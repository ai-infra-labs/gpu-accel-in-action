"""引擎已经生成，为什么推理还会报错？（第8章 · TensorRT 部署）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch、TensorRT、ONNX Runtime、NumPy。
"""
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch


class FeatureRunner:
    def __init__(self, engine_path):
        if trt.__version__.split(".")[0] != "11":
            raise RuntimeError("本例面向 TensorRT 11.x。")
        if not torch.cuda.is_available():
            raise RuntimeError("需要 CUDA 可用的 PyTorch 与 NVIDIA GPU。")

        torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")
        self.stream = torch.cuda.Stream(device=self.device)
        # 保留这些对象，避免仍在使用的 Runtime/Engine 过早销毁。
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            Path(engine_path).read_bytes()
        )
        if self.engine is None:
            raise RuntimeError("引擎加载失败，请核对构建环境与日志。")

        engine = self.engine
        names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
        if names != {"features", "logits"}:
            raise ValueError("本例要求 features 输入和 logits 输出。")
        for name, mode in (("features", trt.TensorIOMode.INPUT),
                           ("logits", trt.TensorIOMode.OUTPUT)):
            if engine.get_tensor_mode(name) != mode:
                raise ValueError(f"{name}: 输入输出方向不符")
            if engine.get_tensor_dtype(name) != trt.float32:
                raise ValueError(f"{name}: 本例仅支持 FP32")
            if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError(f"{name}: 本例要求 GPU 内存")
            if engine.get_tensor_format(name, 0) != trt.TensorFormat.LINEAR:
                raise ValueError(f"{name}: 本例仅支持线性格式")

        self.input_shape = tuple(engine.get_tensor_shape("features"))
        if len(self.input_shape) != 2 or self.input_shape[1] != 256:
            raise ValueError("本例要求 [B, 256] 输入，只有 batch 可以动态。")
        self.dynamic = self.input_shape[0] == -1
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("无法创建执行上下文。")
        if self.dynamic:
            self.min_shape, _, self.max_shape = engine.get_tensor_profile_shape(
                "features", 0
            )
            if not self.context.set_optimization_profile_async(
                0, self.stream.cuda_stream
            ):
                raise RuntimeError("选择 profile 0 失败。")
            self.stream.synchronize()

    def __call__(self, x):
        if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 256:
            raise ValueError("输入必须是 float32 的 [B, 256] 数组。")
        if x.shape[0] < 1 or not np.isfinite(x).all():
            raise ValueError("本例要求非空、有限数值输入。")
        if self.dynamic:
            if not self.min_shape[0] <= x.shape[0] <= self.max_shape[0]:
                raise ValueError("batch 超出 profile 0 的允许范围。")
            if not self.context.set_input_shape("features", tuple(x.shape)):
                raise ValueError("设置实际输入形状失败。")
        elif tuple(x.shape) != self.input_shape:
            raise ValueError(f"固定引擎要求输入形状 {self.input_shape}。")

        output_shape = tuple(self.context.get_tensor_shape("logits"))
        if output_shape != (x.shape[0], 10):
            raise ValueError(f"输出形状未确定或不符合本例约定: {output_shape}")

        with torch.cuda.stream(self.stream):
            # 连续化 CPU 输入；阻塞式搬运不依赖页锁定内存或复制重叠。
            x_gpu = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
            y_gpu = torch.empty(output_shape, device=self.device, dtype=torch.float32)
            for name, tensor in (("features", x_gpu), ("logits", y_gpu)):
                if not self.context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"绑定 {name} 地址失败。")
            try:
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("TensorRT 执行提交失败。")
            finally:
                # 保证退出前等待；输入输出对象在此期间仍然存活。
                self.stream.synchronize()
        return y_gpu.cpu().numpy()


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else "onnx_demo/small_classifier.engine")
    runner = FeatureRunner(path)
    reference = ort.InferenceSession(
        str(path.with_suffix(".onnx")), providers=["CPUExecutionProvider"]
    )
    print("TensorRT:", trt.__version__, "PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))
    rng = np.random.default_rng(42)
    # 动态引擎追加一次 8→1，检查从大 batch 切回小 batch 时
    # 形状是否被重新落实、输出是否被重新查询、地址是否被重新绑定。
    batches = (1, 4, 8, 1) if runner.dynamic else (runner.input_shape[0],)
    for batch in batches:
        for scale in (0.0, 0.1, 1.0):
            x = (rng.standard_normal((batch, 256)) * scale).astype(np.float32)
            expected = reference.run(["logits"], {"features": x})[0]
            actual = runner(x)
            if actual.shape != expected.shape:
                raise AssertionError("参考输出与引擎输出形状不同。")
            if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
                raise AssertionError("输出存在 NaN 或 Inf。")
            error = float(np.max(np.abs(actual - expected)))
            print(f"batch={batch}, scale={scale}, 最大绝对误差={error:.3e}")
            np.testing.assert_allclose(
                actual, expected, atol=1e-5, rtol=1e-4, equal_nan=False
            )
    print("所测输入通过数值核对；不代表分类质量或性能已达标。")


if __name__ == "__main__":
    main()
