"""裁剪不是“切一刀”：用 PyCUDA 讲透 ROI 坐标映射（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import time

import numpy as np
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


KERNEL_CODE = r"""
__global__ void crop_image(
    const float *input,
    float *output,
    int width,
    int height,
    int crop_x,
    int crop_y,
    int crop_width,
    int crop_height)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= crop_width || ty >= crop_height) {
        return;
    }

    int input_x = crop_x + tx;
    int input_y = crop_y + ty;

    if (input_x < 0 || input_x >= width ||
        input_y < 0 || input_y >= height) {
        return;
    }

    int input_index = input_y * width + input_x;
    int output_index = ty * crop_width + tx;

    output[output_index] = input[input_index];
}
"""


def ceil_div(value, divisor):
    """整数除法向上取整。"""
    return (value + divisor - 1) // divisor


def kernel_median_p95(kernel, *args, block, grid, warmup=5, runs=20):
    """CUDA Event 测纯内核时间：warmup 不计时，repeat 取 median/P95。"""
    for _ in range(warmup):
        kernel(*args, block=block, grid=grid)
    cuda.Context.synchronize()

    times = []
    start = cuda.Event()
    end = cuda.Event()
    for _ in range(runs):
        start.record()
        kernel(*args, block=block, grid=grid)
        end.record()
        end.synchronize()
        times.append(start.time_till(end))

    return float(np.median(times)), float(np.percentile(times, 95))


def e2e_median_p95(host_input, host_output, device_input, device_output,
                   kernel, *args, block, grid, warmup=5, runs=20):
    """主机时钟测 H2D + Kernel + D2H 的完整调用窗口（显式同步，
    只用于展示完整调用成本，不作为高性能异步流水线的实现模板）。"""
    for _ in range(warmup):
        cuda.memcpy_htod(device_input, host_input)
        kernel(*args, block=block, grid=grid)
        cuda.memcpy_dtoh(host_output, device_output)
    cuda.Context.synchronize()

    times = []
    for _ in range(runs):
        # 显式同步：让 GPU 侧上个窗口收尾，计时边界明确
        cuda.Context.synchronize()
        t0 = time.perf_counter()
        cuda.memcpy_htod(device_input, host_input)
        kernel(*args, block=block, grid=grid)
        cuda.memcpy_dtoh(host_output, device_output)
        cuda.Context.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.median(times)), float(np.percentile(times, 95))


def validate_roi(
    width,
    height,
    crop_x,
    crop_y,
    crop_width,
    crop_height,
):
    """采用严格语义：ROI 必须完整位于输入图像内。"""
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("裁剪宽度和高度必须大于 0")

    if (
        crop_x < 0
        or crop_y < 0
        or crop_x + crop_width > width
        or crop_y + crop_height > height
    ):
        raise ValueError("裁剪区域超出原始图像边界")


# 编译并获取 CUDA 内核
module = SourceModule(KERNEL_CODE)
crop_kernel = module.get_function("crop_image")

# 生成连续的 float32 单通道图像
width, height = 512, 512
rng = np.random.default_rng(42)
input_image = rng.random((height, width), dtype=np.float32)
assert input_image.flags["C_CONTIGUOUS"], "内核假设行优先连续存储"

# ROI 左上角与尺寸
crop_x, crop_y = 100, 100
crop_width, crop_height = 100, 100

validate_roi(
    width,
    height,
    crop_x,
    crop_y,
    crop_width,
    crop_height,
)

# 输出是独立、连续的 100×100 数组
gpu_crop = np.empty((crop_height, crop_width), dtype=np.float32)

# 显式分配设备内存并上传输入
input_gpu = cuda.mem_alloc(input_image.nbytes)
output_gpu = cuda.mem_alloc(gpu_crop.nbytes)
cuda.memcpy_htod(input_gpu, input_image)

# 256 线程/Block：二维映射直观、又是 32 的倍数；教学起点，而非所有 GPU 上的最优配置
block = (16, 16, 1)
grid = (
    ceil_div(crop_width, block[0]),
    ceil_div(crop_height, block[1]),
)

# 统一的内核参数，预热、Kernel 计时与 E2E 计时共用
kernel_args = (
    input_gpu,
    output_gpu,
    np.int32(width),
    np.int32(height),
    np.int32(crop_x),
    np.int32(crop_y),
    np.int32(crop_width),
    np.int32(crop_height),
)

# CUDA Event 测纯内核时间：warmup×5 + repeat×20，取 median 与 P95
kernel_median, kernel_p95 = kernel_median_p95(
    crop_kernel, *kernel_args, block=block, grid=grid
)

# 设备到主机复制
cuda.memcpy_dtoh(gpu_crop, output_gpu)

# NumPy 基准结果；copy() 明确物化为独立连续数组
cpu_crop = input_image[
    crop_y:crop_y + crop_height,
    crop_x:crop_x + crop_width,
].copy()

# 纯复制没有浮点计算，应逐字节一致；array_equal 比 allclose 语义更强
ok = np.array_equal(gpu_crop, cpu_crop)

# 打印环境信息，便于复现
device = cuda.Device(0)
print(f"GPU: {device.name()}")
print(f"PyCUDA: {pycuda.VERSION_TEXT}, NumPy: {np.__version__}")
print(
    f"dtype: {input_image.dtype}, Input: {width}x{height}, "
    f"ROI: {crop_width}x{crop_height} @ ({crop_x},{crop_y})"
)
print(f"Block: {block}, Grid: {grid}")
print(f"Kernel latency (median / P95): {kernel_median:.3f} / {kernel_p95:.3f} ms")
print("结果正确（逐字节一致）:", ok)
assert ok

# 端到端时间：H2D + Kernel + D2H（主机时钟，warmup×5 + repeat×20）
e2e_median, e2e_p95 = e2e_median_p95(
    input_image,
    gpu_crop,
    input_gpu,
    output_gpu,
    crop_kernel,
    *kernel_args,
    block=block,
    grid=grid,
)

print(f"End-to-end (H2D+Kernel+D2H, median / P95): {e2e_median:.3f} / {e2e_p95:.3f} ms")
