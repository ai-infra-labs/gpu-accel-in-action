"""
PyTorch → ONNX → TensorRT 10.x 对照实验（配套正文完整代码）

三条路径对比：PyTorch GPU FP32 / TensorRT FP32 / TensorRT FP16
只统计 GPU 计算时间，H2D/D2H 不混入；误差对齐 PyTorch FP32 输出。

在有 NVIDIA GPU 的机器上运行：
    python trt_mlp_benchmark.py
依赖：torch、onnx、numpy、tensorrt(10.x)、pycuda
"""

import os

import numpy as np
import onnx
import pycuda.autoprimaryctx          # 自动创建并激活主 CUDA 上下文
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn as nn


# ------------------------- 1. 示例模型 -------------------------
class MLP(nn.Module):
    """商品打分头（演示用）：权重随机初始化，不代表已训练的业务模型。"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 512),
        )

    def forward(self, x):
        return self.layers(x)


def export_onnx(model, onnx_path, sample_input):
    """导出 batch 维动态的 ONNX 并做结构检查。"""
    model.eval()
    torch.onnx.export(
        model, sample_input, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    onnx.checker.check_model(onnx.load(onnx_path))   # 结构不合法会在此抛错


# ------------------------- 2. 构建 engine -------------------------
def build_engine(onnx_path, precision, max_batch=64):
    """构建并保存指定精度的 engine，返回反序列化后的对象与 plan 路径。"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TensorRT 10.x：显式 batch 是唯一模式，8.x 的 EXPLICIT_BATCH 标志已移除
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(str(parser.get_error(i))
                               for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX 解析失败：\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

    profile = builder.create_optimization_profile()   # 动态形状合同：min/opt/max
    if not profile.set_shape("input", (1, 1024), (16, 1024), (max_batch, 1024)):
        raise RuntimeError("设置动态输入 profile 失败：请检查输入张量名")
    config.add_optimization_profile(profile)

    # 关闭 TF32，保证与 PyTorch 基线同为 IEEE FP32 口径（详见正文实验设计）
    config.clear_flag(trt.BuilderFlag.TF32)

    if precision == "fp16":
        # platform_has_fast_fp16 在部分 10.x 构建中已从 Builder 移除，缺失时视为可用
        if not getattr(builder, "platform_has_fast_fp16", True):
            raise RuntimeError("当前 GPU 不支持快速 FP16，无法完成该对照")
        config.set_flag(trt.BuilderFlag.FP16)   # “允许”，不是“强制”，见正文
    elif precision != "fp32":
        raise ValueError("precision 仅支持 'fp32' 或 'fp16'")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine 构建失败（请查看 WARNING 日志）")

    engine_path = f"mlp_{precision}.plan"
    with open(engine_path, "wb") as f:
        f.write(serialized)

    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT engine 反序列化失败")
    return engine, engine_path


# ------------------------- 3. 可复用推理器 -------------------------
class TensorRTRunner:
    """封装执行上下文、CUDA 流与设备内存，按名称绑定张量（10.x 风格）。"""

    def __init__(self, engine, input_shape):
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()
        self.input_shape = tuple(input_shape)

        # 若 ONNX 里改了名字，用 engine.get_tensor_name(i) 遍历确认
        self.input_name = "input"
        self.output_name = "output"
        if not self.context.set_input_shape(self.input_name, self.input_shape):
            raise RuntimeError(f"输入形状未命中 profile：{self.input_shape}")

        out_dims = self.context.get_tensor_shape(self.output_name)
        if any(d <= 0 for d in out_dims):
            raise RuntimeError(f"输出形状无效：{tuple(out_dims)}")
        self.output_shape = tuple(out_dims)          # 形状由输入动态推导出来

        in_dtype = trt.nptype(engine.get_tensor_dtype(self.input_name))
        out_dtype = trt.nptype(engine.get_tensor_dtype(self.output_name))
        self.host_input = np.empty(self.input_shape, dtype=in_dtype)
        self.host_output = np.empty(self.output_shape, dtype=out_dtype)
        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)

        # TensorRT 10.x：地址注册与执行分离，不再传 bindings 数组
        self.context.set_tensor_address(self.input_name, int(self.device_input))
        self.context.set_tensor_address(self.output_name, int(self.device_output))

    def copy_input(self, input_array):
        if tuple(input_array.shape) != self.input_shape:
            raise ValueError(f"输入应为 {self.input_shape}，实际 {input_array.shape}")
        self.host_input[...] = input_array
        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)

    def enqueue(self):
        if not self.context.execute_async_v3(stream_handle=self.stream.handle):
            raise RuntimeError("TensorRT 推理执行失败")

    def infer(self, input_array):
        """完整推理一次：上卡 → 执行 → 下卡 → 同步 → 返回主机端结果。"""
        self.copy_input(input_array)
        self.enqueue()
        cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()
        return self.host_output.copy()

    def benchmark_gpu_compute(self, input_array, warmup=20, runs=200):
        """只统计 GPU 上重复执行 engine 的时间，不含主机设备间拷贝。"""
        self.copy_input(input_array)
        self.stream.synchronize()
        for _ in range(warmup):
            self.enqueue()
        self.stream.synchronize()

        start, end = cuda.Event(), cuda.Event()
        start.record(self.stream)
        for _ in range(runs):
            self.enqueue()
        end.record(self.stream)
        end.synchronize()
        return start.time_till(end) / runs          # 单位：毫秒 / 次


# ------------------------- 4. PyTorch 基线与误差 -------------------------
def benchmark_torch_gpu(model, input_gpu, warmup=20, runs=200):
    """用 CUDA Event 统计 PyTorch eager 的 GPU 计算时间。"""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_gpu)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(runs):
            _ = model(input_gpu)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / runs       # 单位：毫秒 / 次


def error_stats(reference, candidate):
    """返回最大绝对误差与平均绝对误差（对齐 reference）。"""
    diff = np.abs(reference - candidate)
    return float(diff.max()), float(diff.mean())


def register_result(name, ms, ref, out):
    """打印单条路径的完整指标（速度 + 误差 + allclose）。"""
    max_err, mean_err = error_stats(ref, out)
    print(f"\n[{name}]")
    print(f"  平均 GPU 计算时间：{ms:.4f} ms/次")
    print(f"  最大绝对误差：{max_err:.6e}")
    print(f"  平均绝对误差：{mean_err:.6e}")
    print(f"  allclose(rtol=1e-2, atol=1e-2)："
          f"{np.allclose(ref, out, rtol=1e-2, atol=1e-2)}")
    return max_err, mean_err


# ------------------------- 5. 对照实验入口 -------------------------
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("本脚本需要在 CUDA 版 PyTorch 与 NVIDIA GPU 上运行")

    torch.manual_seed(2026)
    np.random.seed(2026)

    # 显式关闭 TF32，确保 PyTorch 与 TensorRT 的 FP32 基线口径一致（正文实验设计）
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")
    model = MLP().to(device).eval()

    batch_size = 16
    input_shape = (batch_size, 1024)
    input_cpu = np.random.randn(*input_shape).astype(np.float32)
    input_gpu = torch.from_numpy(input_cpu).to(device)

    onnx_path = "mlp.onnx"
    export_onnx(model, onnx_path, input_gpu)

    # 基线：PyTorch GPU FP32
    with torch.no_grad():
        torch_output = model(input_gpu).float().cpu().numpy()
    torch_ms = benchmark_torch_gpu(model, input_gpu)
    print(f"PyTorch GPU 平均计算时间：{torch_ms:.4f} ms/次（基线 1.00x）")

    # 对照：TensorRT FP32 / FP16
    for precision in ("fp32", "fp16"):
        engine, engine_path = build_engine(onnx_path, precision)
        runner = TensorRTRunner(engine, input_shape)
        trt_output = runner.infer(input_cpu)
        trt_ms = runner.benchmark_gpu_compute(input_cpu)

        register_result(f"TensorRT {precision.upper()}", trt_ms,
                        torch_output, trt_output)
        print(f"  相对基线加速比：{torch_ms / trt_ms:.2f}x")
        print(f"  engine 文件：{os.path.getsize(engine_path) / 1024:.1f} KiB")
        print(f"  执行上下文设备内存需求："
              f"{int(engine.device_memory_size) / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
