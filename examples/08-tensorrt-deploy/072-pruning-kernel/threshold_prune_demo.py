"""给权重做剪枝，CUDA 到底加速了哪一步？（第8章 · TensorRT 部署）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import time
import numpy as np
import pycuda
import pycuda.autoinit  # 本实验独立建立上下文，不混用其他框架的指针。
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


CUDA_SOURCE = r"""
#include <math.h>
__global__ void threshold_prune(
    const float *input, float *output, int n, float threshold)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (unsigned int)n) {
        float w = input[i];
        output[i] = fabsf(w) > threshold ? w : 0.0f;
    }
}
"""


def cpu_reference(source, threshold):
    return np.where(np.abs(source) > threshold, source, np.float32(0))


def measure_wall(call, gpu=False, warmup=10, samples=50):
    for _ in range(warmup):
        call()
        if gpu:
            cuda.Context.synchronize()
    times = []
    for _ in range(samples):
        if gpu:
            cuda.Context.synchronize()
        start = time.perf_counter()
        call()
        if gpu:
            cuda.Context.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return {"samples": samples, "p50_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95, method="linear"))}


def main():
    # 编译/加载位于采样之外；缓存可能影响这一步，不能称为冷编译测量。
    start = time.perf_counter()
    module = SourceModule(CUDA_SOURCE)
    kernel = module.get_function("threshold_prune")
    print("源码编译/加载墙钟秒数:", time.perf_counter() - start)
    print("PyCUDA:", pycuda.VERSION_TEXT)
    print("GPU:", cuda.Context.get_device().name())
    threshold = np.float32(0.125)
    rng = np.random.default_rng(42)

    # 小规模、线程块边界、示例网络规模、大数组分别观察。
    for n in (0, 1, 255, 256, 257, 41600, 1000003, 4194304):
        source = rng.standard_normal(n).astype(np.float32)
        edges = np.array([-1, -threshold, -0.0, 0.0, threshold, 1], np.float32)
        source[:min(n, len(edges))] = edges[:min(n, len(edges))]
        expected = cpu_reference(source, threshold)
        if n == 0:
            assert expected.size == 0
            print("n=0：按约定返回空数组，不分配 GPU 内存或启动 kernel。")
            continue

        # 设备数组均为本次独立 PyCUDA 上下文中的连续 FP32 数组。
        d_input = gpuarray.to_gpu(source)
        d_output = gpuarray.empty_like(d_input)
        d_high = gpuarray.empty_like(d_input)
        d_zero = gpuarray.zeros_like(d_input)
        host_output = np.empty_like(source)
        grid = ((n + 255) // 256, 1, 1)

        def launch():
            kernel(d_input.gpudata, d_output.gpudata, np.int32(n), threshold,
                   block=(256, 1, 1), grid=grid)

        def high_level():
            criterion = abs(d_input) - threshold
            gpuarray.if_positive(criterion, d_input, d_zero, out=d_high)

        def round_trip():
            # 同步复制，包含传输但排除缓冲区分配和编译。
            cuda.memcpy_htod(d_input.gpudata, source)
            launch()
            cuda.memcpy_dtoh(host_output, d_output.gpudata)

        # 先正确性、后性能。get() 只用于检查，不放进驻留操作计时。
        launch()
        high_level()
        cuda.Context.synchronize()
        np.testing.assert_array_equal(d_output.get(), expected)
        np.testing.assert_array_equal(d_high.get(), expected)
        round_trip()
        np.testing.assert_array_equal(host_output, expected)

        print("\nn:", n, "CPU/GPU结果在本组输入上相同")
        print("CPU参考（含NumPy临时数组）:",
              measure_wall(lambda: cpu_reference(source, threshold)))
        print("GPU数组表达式（驻留、含临时数组与同步）:",
              measure_wall(high_level, gpu=True))
        print("自定义kernel（驻留、含提交与同步）:",
              measure_wall(launch, gpu=True))
        print("自定义kernel往返（含H2D/D2H）:",
              measure_wall(round_trip, gpu=True))


if __name__ == "__main__":
    main()
