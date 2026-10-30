"""两个 Kernel，为什么可以合成一个？（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import time

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


KERNEL_CODE = r"""
extern "C" __global__ void preprocess_batch(
    const float *__restrict__ input,
    float *__restrict__ output,
    int batch_size,
    int height,
    int width,
    int crop_size,
    float mean,
    float inv_std)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (b >= batch_size || tx >= crop_size || ty >= crop_size) {
        return;
    }

    int offset_x = (width - crop_size) / 2;
    int offset_y = (height - crop_size) / 2;
    int crop_x = offset_x + tx;
    int crop_y = offset_y + ty;

    size_t input_index =
        ((size_t)b * height + crop_y) * width + crop_x;

    size_t output_index =
        ((size_t)b * crop_size + ty) * crop_size + tx;

    float pixel = input[input_index];
    output[output_index] = (pixel - mean) * inv_std;
}
"""


def ceil_div(value, divisor):
    """整数除法向上取整。"""
    return (value + divisor - 1) // divisor


def validate_args(batch_size, height, width, crop_size, mean, stddev):
    """在启动 GPU 前统一校验标量参数。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    if height <= 0 or width <= 0:
        raise ValueError("height 和 width 必须大于 0")

    if crop_size <= 0 or crop_size > min(height, width):
        raise ValueError("crop_size 必须位于 (0, min(height, width)]")

    if not np.isfinite(mean):
        raise ValueError("mean 必须是有限数")

    if not np.isfinite(stddev) or stddev <= 0:
        raise ValueError("stddev 必须是大于 0 的有限数")


# 示例参数：8 张 256×256 单通道 float32 图像，中心裁为 224×224
batch_size = 8
height, width = 256, 256
crop_size = 224
mean = 0.5
stddev = 0.25

validate_args(
    batch_size,
    height,
    width,
    crop_size,
    mean,
    stddev,
)

# 输入数据已经位于 0～1；mean/stddev 也基于相同尺度
rng = np.random.default_rng(42)
input_batch = rng.random(
    (batch_size, height, width),
    dtype=np.float32,
)

gpu_output = np.empty(
    (batch_size, crop_size, crop_size),
    dtype=np.float32,
)

# KERNEL_CODE 已显式声明 extern "C"，因此关闭 SourceModule 的自动包装
module = SourceModule(KERNEL_CODE, no_extern_c=True)
preprocess_kernel = module.get_function("preprocess_batch")

# 分配显存；显存分配不放入纯 Kernel 计时窗口
input_gpu = cuda.mem_alloc(input_batch.nbytes)
output_gpu = cuda.mem_alloc(gpu_output.nbytes)

block = (16, 16, 1)
grid = (
    ceil_div(crop_size, block[0]),
    ceil_div(crop_size, block[1]),
    batch_size,
)

inv_std = np.float32(1.0 / stddev)

# 上传输入
cuda.memcpy_htod(input_gpu, input_batch)

# 预热，避免首次启动开销污染正式测量
preprocess_kernel(
    input_gpu,
    output_gpu,
    np.int32(batch_size),
    np.int32(height),
    np.int32(width),
    np.int32(crop_size),
    np.float32(mean),
    inv_std,
    block=block,
    grid=grid,
)
cuda.Context.synchronize()

# 使用 CUDA Event 测量纯 GPU Kernel 时间
start_event = cuda.Event()
end_event = cuda.Event()

start_event.record()
preprocess_kernel(
    input_gpu,
    output_gpu,
    np.int32(batch_size),
    np.int32(height),
    np.int32(width),
    np.int32(crop_size),
    np.float32(mean),
    inv_std,
    block=block,
    grid=grid,
)
end_event.record()
end_event.synchronize()

kernel_ms = start_event.time_till(end_event)

# 下载最终输出
cuda.memcpy_dtoh(gpu_output, output_gpu)

# NumPy 参考实现：同样的中心偏移、同样的输入尺度
offset_x = (width - crop_size) // 2
offset_y = (height - crop_size) // 2

cpu_reference = input_batch[
    :,
    offset_y:offset_y + crop_size,
    offset_x:offset_x + crop_size,
]
cpu_reference = (cpu_reference - mean) / stddev

print(f"Block: {block}")
print(f"Grid:  {grid}")
print(f"Kernel 时间: {kernel_ms:.3f} ms")
print(
    "结果正确:",
    np.allclose(gpu_output, cpu_reference, rtol=1e-5, atol=1e-6),
)

# 端到端示例：H2D + Kernel + D2H
e2e_start = time.perf_counter()
cuda.memcpy_htod(input_gpu, input_batch)
preprocess_kernel(
    input_gpu,
    output_gpu,
    np.int32(batch_size),
    np.int32(height),
    np.int32(width),
    np.int32(crop_size),
    np.float32(mean),
    inv_std,
    block=block,
    grid=grid,
)
cuda.memcpy_dtoh(gpu_output, output_gpu)
e2e_ms = (time.perf_counter() - e2e_start) * 1000

print(f"端到端时间: {e2e_ms:.3f} ms")
