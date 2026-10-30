"""推理只要 1 毫秒，应用为什么还在等？（第8章 · TensorRT 部署）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch、TensorRT、ONNX Runtime、NumPy。
"""
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch

from run_engine import FeatureRunner


def measure_wall(call, batch, warmup=50, samples=500):
    for _ in range(warmup):
        call()  # call 返回时 CPU 输出已经可读。
    times_ms = []
    total_start = time.perf_counter()
    for _ in range(samples):
        start = time.perf_counter()
        call()
        times_ms.append((time.perf_counter() - start) * 1000)
    total_seconds = time.perf_counter() - total_start
    return {
        "samples": samples,
        "mean_ms": float(np.mean(times_ms)),
        "p50_ms": float(np.percentile(times_ms, 50, method="linear")),
        "p95_ms": float(np.percentile(times_ms, 95, method="linear")),
        # 总循环墙钟包含计时和记录开销；这是串行实验完成速率。
        "serial_samples_per_second": batch * samples / total_seconds,
    }


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else "onnx_demo/small_classifier.engine")
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    runner = FeatureRunner(path)
    ctx, stream = runner.context, runner.stream
    x_np = np.random.default_rng(42).standard_normal((batch, 256)).astype(np.float32)

    # 先核对正确性，并落实当前 batch；这些步骤不在性能计时内。
    session = ort.InferenceSession(
        str(path.with_suffix(".onnx")), providers=["CPUExecutionProvider"]
    )
    expected = session.run(["logits"], {"features": x_np})[0]
    actual = runner(x_np)
    np.testing.assert_allclose(
        actual, expected, atol=1e-5, rtol=1e-4, equal_nan=False
    )

    x_host = torch.empty((batch, 256), dtype=torch.float32, pin_memory=True)
    y_host = torch.empty((batch, 10), dtype=torch.float32, pin_memory=True)
    x_host.copy_(torch.from_numpy(x_np))
    with torch.cuda.stream(stream):
        x_buffer = torch.empty_like(x_host, device=runner.device)
        y_buffer = torch.empty_like(y_host, device=runner.device)

    def bind(x_gpu, y_gpu):
        for name, tensor in (("features", x_gpu), ("logits", y_gpu)):
            if not ctx.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"绑定 {name} 失败")

    def enqueue():
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("提交执行失败")

    def call(reuse):
        with torch.cuda.stream(stream):
            if reuse:
                x_gpu, y_gpu = x_buffer, y_buffer
            else:
                x_gpu = torch.empty_like(x_buffer)
                y_gpu = torch.empty_like(y_buffer)
            try:
                x_gpu.copy_(x_host, non_blocking=True)
                bind(x_gpu, y_gpu)
                enqueue()
                y_host.copy_(y_gpu, non_blocking=True)
            finally:
                stream.synchronize()  # 等待复制与计算，所有缓冲区仍存活。

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("Engine:", path, "batch:", batch, "TensorRT:", trt.__version__)
    for reuse in (False, True):
        call(reuse)
        # 两条被测路径都先验收；检查和打印不放进采样区间。
        np.testing.assert_allclose(
            y_host.numpy(), expected, atol=1e-5, rtol=1e-4, equal_nan=False
        )
        result = measure_wall(lambda: call(reuse), batch)
        print("复用缓冲区" if reuse else "每次创建张量", result)

    # 输入留在 GPU，地址固定；事件区间内不做 H2D、D2H 或新张量分配。
    call(True)
    bind(x_buffer, y_buffer)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # 先初始化事件，避免把事件首次创建混入正式区间。
    start_event.record(stream)
    end_event.record(stream)
    end_event.synchronize()
    group_means = []
    repeats = 200
    try:
        for _ in range(10):
            start_event.record(stream)
            for _ in range(repeats):
                enqueue()
            end_event.record(stream)
            end_event.synchronize()
            group_means.append(start_event.elapsed_time(end_event) / repeats)
    finally:
        stream.synchronize()
    print("事件法：10组、每组200次，平均流区间 ms/次:", float(np.mean(group_means)))


if __name__ == "__main__":
    main()
