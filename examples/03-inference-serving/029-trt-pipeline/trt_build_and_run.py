"""模型上了 GPU，一次请求的时间都花在哪？一次 TensorRT 对照实验（第3章 · 推理服务与部署）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch、PyCUDA、TensorRT、ONNX、NumPy。
"""
import os
import numpy as np
import onnx
import pycuda.autoprimaryctx  # 让 PyCUDA 与 PyTorch 使用 CUDA primary context
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn as nn
from torch.export import Dim


class ClassifierHead(nn.Module):
    """示例分类头：输入 1024 维特征，输出 512 个类别分数。"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
        )

    def forward(self, x):
        return self.layers(x)


def export_onnx(model, path, sample):
    """导出带动态 batch 维度的 ONNX 模型。"""

    torch.onnx.export(
        model,
        (sample,),
        path,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
        dynamic_shapes={
            "x": {0: Dim("batch", min=1, max=64)}
        },
        opset_version=17,
    )

    # 检查 ONNX 文件结构；通过检查不代表 TensorRT 一定支持所有算子。
    onnx.checker.check_model(onnx.load(path))


def build_engine(onnx_path, precision, batch_max=64):
    """使用 TensorRT 8.x API 构建指定精度的 engine。"""

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # 显式 batch：batch 是输入张量形状的一部分。
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        parsed = parser.parse(f.read())

    if not parsed:
        errors = "\n".join(
            str(parser.get_error(i)) for i in range(parser.num_errors)
        )
        raise RuntimeError(f"ONNX 解析失败：\n{errors}")

    config = builder.create_builder_config()
    # TensorRT 8.6：设置构建阶段 WORKSPACE 内存池上限为 1 GiB。
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    profile = builder.create_optimization_profile()
    ok = profile.set_shape(
        "input",
        (1, 1024),          # 最小 batch
        (16, 1024),         # 常用 batch，builder 重点优化
        (batch_max, 1024),  # 最大 batch
    )
    if not ok:
        raise RuntimeError("动态输入 profile 配置失败")
    config.add_optimization_profile(profile)

    # 让 FP32 基线不使用 TF32；FP16 模式则允许 builder 选择 FP16 tactic。
    config.clear_flag(trt.BuilderFlag.TF32)
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("当前 GPU 不支持快速 FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision != "fp32":
        raise ValueError("precision 只能是 fp32 或 fp16")

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine 构建失败")

    plan_path = f"classifier_{precision}.plan"
    with open(plan_path, "wb") as f:
        f.write(plan)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("TensorRT engine 反序列化失败")

    return engine, plan_path


class Runner:
    """复用执行上下文、CUDA 流和显存缓冲区。"""

    def __init__(self, engine, shape):
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()
        self.shape = tuple(shape)

        self.input_index = engine.get_binding_index("input")
        self.output_index = engine.get_binding_index("output")

        # 动态输入先设置本次实际形状。
        if not self.context.set_binding_shape(self.input_index, self.shape):
            raise RuntimeError(f"输入形状不在 profile 范围内：{self.shape}")

        self.output_shape = tuple(
            self.context.get_binding_shape(self.output_index)
        )
        if any(d <= 0 for d in self.output_shape):
            raise RuntimeError(f"输出形状无效：{self.output_shape}")

        self.host_input = np.empty(self.shape, dtype=np.float32)
        self.host_output = np.empty(self.output_shape, dtype=np.float32)
        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)

        # 地址列表必须按 binding 索引排列。
        self.bindings = [0] * engine.num_bindings
        self.bindings[self.input_index] = int(self.device_input)
        self.bindings[self.output_index] = int(self.device_output)

    def enqueue(self):
        ok = self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )
        if not ok:
            raise RuntimeError("TensorRT 推理提交失败")

    def infer(self, x):
        """完整执行一次：主机输入 → GPU 计算 → 主机输出。"""

        self.host_input[...] = x
        cuda.memcpy_htod_async(
            self.device_input, self.host_input, self.stream
        )
        self.enqueue()
        cuda.memcpy_dtoh_async(
            self.host_output, self.device_output, self.stream
        )
        self.stream.synchronize()
        return self.host_output.copy()

    def benchmark_compute(self, x, warmup=20, runs=200):
        """只计 GPU 计算，不计输入输出拷贝。"""

        self.host_input[...] = x
        cuda.memcpy_htod_async(
            self.device_input, self.host_input, self.stream
        )
        self.stream.synchronize()

        for _ in range(warmup):
            self.enqueue()
        self.stream.synchronize()

        start = cuda.Event()
        end = cuda.Event()
        start.record(self.stream)

        for _ in range(runs):
            self.enqueue()

        end.record(self.stream)
        end.synchronize()
        return start.time_till(end) / runs  # 毫秒/次


def benchmark_torch(model, x, warmup=20, runs=200):
    """测量 PyTorch GPU 前向计算的平均时间。"""

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(runs):
            model(x)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / runs


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("请在有 NVIDIA GPU 的 CUDA 环境中运行")

    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")
    model = ClassifierHead().eval()
    batch = 16
    x_np = np.random.randn(batch, 1024).astype(np.float32)

    # 先用 CPU 模型和 CPU 示例输入导出 ONNX，再把同一组权重搬到 GPU。
    onnx_path = "classifier.onnx"
    export_onnx(model, onnx_path, torch.from_numpy(x_np))
    model = model.to(device).eval()
    x_torch = torch.from_numpy(x_np).to(device)

    with torch.no_grad():
        reference = model(x_torch).float().cpu().numpy()

    torch_ms = benchmark_torch(model, x_torch)
    print(f"PyTorch GPU：{torch_ms:.4f} ms/次")

    for precision in ("fp32", "fp16"):
        engine, plan_path = build_engine(onnx_path, precision)
        runner = Runner(engine, (batch, 1024))

        # infer 用于取回输出；benchmark_compute 单独测 GPU 计算时间。
        output = runner.infer(x_np)
        trt_ms = runner.benchmark_compute(x_np)

        abs_error = np.abs(reference - output)
        tolerance = 1e-4 if precision == "fp32" else 1e-2
        same = np.allclose(
            reference, output, rtol=tolerance, atol=tolerance
        )

        # 这些是可统计的内存组成，不是整个进程的显存峰值。
        plan_bytes = os.path.getsize(plan_path)
        io_device_bytes = runner.host_input.nbytes + runner.host_output.nbytes
        context_bytes = int(engine.device_memory_size)

        print(f"\nTensorRT {precision.upper()}")
        print(f"GPU 计算时间：{trt_ms:.4f} ms/次")
        print(f"相对 PyTorch：{torch_ms / trt_ms:.2f}x")
        print(f"最大绝对误差：{abs_error.max():.6e}")
        print(f"平均绝对误差：{abs_error.mean():.6e}")
        print(f"容差检查：{'通过' if same else '未通过'}")
        print(f"engine 文件：{plan_bytes / 1024:.1f} KiB")
        print(f"context 设备内存需求：{context_bytes / 1024:.1f} KiB")
        print(f"输入输出 GPU 缓冲区：{io_device_bytes / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
