"""一个线程处理一个像素：用 PyCUDA 手写图像水平翻转（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import time

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


# CUDA C 内核：一个线程生成一个水平翻转后的输出像素
KERNEL_CODE = r"""
__global__ void flip_horizontal(
    const float *input,
    float *output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int src_x = width - 1 - x;
    int dst_index = y * width + x;
    int src_index = y * width + src_x;

    output[dst_index] = input[src_index];
}
"""


def ceil_div(value, divisor):
    """对正整数执行除法并向上取整。"""
    return (value + divisor - 1) // divisor


def benchmark_kernel(kernel, *args, block, grid, warmup=5, repeat=20):
    """非计时的预热 + 连续测量，返回 (median_ms, p95_ms)，单位毫秒。"""
    # warmup 全程不计时，避免把首次执行的额外开销混入正式测量
    for _ in range(warmup):
        kernel(*args, block=block, grid=grid)
    cuda.Context.synchronize()

    times = []
    start = cuda.Event()
    end = cuda.Event()
    for _ in range(repeat):
        start.record()
        kernel(*args, block=block, grid=grid)
        end.record()
        end.synchronize()
        times.append(start.time_till(end))

    return float(np.median(times)), float(np.percentile(times, 95))


# 编译 CUDA 代码并获取内核函数
module = SourceModule(KERNEL_CODE)
flip_kernel = module.get_function("flip_horizontal")

# 生成连续的 float32 灰度图像
width, height = 4096, 4096
rng = np.random.default_rng(42)
input_image = rng.random((height, width), dtype=np.float32)
gpu_result = np.empty_like(input_image)

# 分配输入、输出显存；分配本身不放入内核计时窗口
input_gpu = cuda.mem_alloc(input_image.nbytes)
output_gpu = cuda.mem_alloc(gpu_result.nbytes)

block = (16, 16, 1)
grid = (
    ceil_div(width, block[0]),
    ceil_div(height, block[1]),
)

# 统一的内核参数，预热与正式测量共用
kernel_args = (
    input_gpu,
    output_gpu,
    np.int32(width),
    np.int32(height),
)

# H2D 传输不属于内核计时；随后的 benchmark_kernel 自带 warmup×5 非计时运行，
# 首次执行相关运行时开销不会进入正式测量
cuda.memcpy_htod(input_gpu, input_image)

# CUDA Event 测量：warmup×5 + repeat×20，取 median 与 P95
median_ms, p95_ms = benchmark_kernel(
    flip_kernel,
    *kernel_args,
    block=block,
    grid=grid,
    warmup=5,
    repeat=20,
)

# 将 GPU 结果复制回主机
cuda.memcpy_dtoh(gpu_result, output_gpu)

# 使用 NumPy 作为参考实现
cpu_reference = np.fliplr(input_image)

# effective bandwidth：按逻辑读写字节数 / 内核时间，这不是 DRAM 带宽
bytes_moved = input_image.nbytes + gpu_result.nbytes
bandwidth_gbps = bytes_moved / (median_ms / 1000) / 1e9

print(f"Grid: {grid}, Block: {block}")
print(f"Kernel 时间（median / P95）: {median_ms:.3f} / {p95_ms:.3f} ms")
print(f"Effective bandwidth（逻辑字节）: {bandwidth_gbps:.2f} GB/s")
print("结果正确:", np.allclose(gpu_result, cpu_reference))

# 单独演示一次端到端计时：H2D + Kernel + D2H（主机时钟，仅作演示）
e2e_start = time.perf_counter()
cuda.memcpy_htod(input_gpu, input_image)
flip_kernel(*kernel_args, block=block, grid=grid)
cuda.memcpy_dtoh(gpu_result, output_gpu)
e2e_ms = (time.perf_counter() - e2e_start) * 1000

print(f"端到端时间（H2D + Kernel + D2H）: {e2e_ms:.3f} ms")
