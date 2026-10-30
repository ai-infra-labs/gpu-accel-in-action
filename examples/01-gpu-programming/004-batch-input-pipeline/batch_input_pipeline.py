"""Kernel 已经很快，GPU 为什么还在等？用 PyCUDA 搭建批量数据流水线（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import time

import numpy as np
import pycuda.autoinit  # 创建并管理当前进程的 CUDA Context
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


KERNEL_CODE = r"""
#include <stddef.h>

__global__ void preprocess_batch(
    const float *__restrict__ input,
    float *__restrict__ output,
    int batch_size,
    int height,
    int width,
    int crop_size,
    float mean,
    float inv_std)
{
    const int tx = threadIdx.x + blockIdx.x * blockDim.x;
    const int ty = threadIdx.y + blockIdx.y * blockDim.y;
    const int b = blockIdx.z;

    if (b >= batch_size || tx >= crop_size || ty >= crop_size) {
        return;
    }

    const int offset_x = (width  - crop_size) / 2;
    const int offset_y = (height - crop_size) / 2;
    const int crop_x = offset_x + tx;
    const int crop_y = offset_y + ty;

    const size_t input_idx =
        ((size_t)b * height + crop_y) * width + crop_x;
    const size_t output_idx =
        ((size_t)b * crop_size + ty) * crop_size + tx;

    output[output_idx] = (input[input_idx] - mean) * inv_std;
}
"""


def ceil_div(value, divisor):
    """整数向上取整，确保 Grid 覆盖全部输出像素。"""
    return (value + divisor - 1) // divisor


def fill_batch(destination, rng):
    """
    模拟真实的数据读取与解码阶段。

    生产环境可以把这里替换成：
    读取文件 -> 解码 -> resize/颜色转换 -> 写入 destination。
    关键是最终数据直接落入复用的 pinned buffer，避免额外中间副本。
    """
    rng.random(destination.shape, dtype=np.float32, out=destination)


def cpu_reference(input_batch, crop_size, mean, stddev):
    """使用 NumPy 生成同一数学定义下的参考结果。"""
    _, height, width = input_batch.shape
    offset_y = (height - crop_size) // 2
    offset_x = (width - crop_size) // 2
    cropped = input_batch[
        :,
        offset_y : offset_y + crop_size,
        offset_x : offset_x + crop_size,
    ]
    return (cropped - mean) / stddev


def main():
    # ------------------------- 参数与合法性检查 -------------------------
    batch_size = 32
    height, width = 256, 256
    crop_size = 224
    mean = np.float32(0.5)
    stddev = np.float32(0.25)
    inv_std = np.float32(1.0 / stddev)

    if crop_size > min(height, width):
        raise ValueError("crop_size 不能大于原图的高或宽")
    if not np.isfinite(stddev) or stddev <= 0:
        raise ValueError("stddev 必须是有限的正数")

    input_shape = (batch_size, height, width)
    output_shape = (batch_size, crop_size, crop_size)
    input_nbytes = int(np.prod(input_shape)) * np.dtype(np.float32).itemsize
    output_nbytes = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize

    # ------------------------------ 编译内核 ------------------------------
    module = SourceModule(KERNEL_CODE)
    preprocess_batch = module.get_function("preprocess_batch")

    block = (16, 16, 1)
    grid = (
        ceil_div(crop_size, block[0]),
        ceil_div(crop_size, block[1]),
        batch_size,
    )

    # ------------------------------ 双缓冲 ------------------------------
    buffer_count = 2
    streams = [cuda.Stream() for _ in range(buffer_count)]

    # 两块页锁定主机内存：支持异步 H2D，并避免循环中反复申请
    host_inputs = [
        cuda.pagelocked_empty(input_shape, dtype=np.float32)
        for _ in range(buffer_count)
    ]

    # 两组独立设备输入/输出，防止不同 Stream 读写同一块缓冲区
    device_inputs = [
        cuda.mem_alloc(input_nbytes)
        for _ in range(buffer_count)
    ]
    device_outputs = [
        cuda.mem_alloc(output_nbytes)
        for _ in range(buffer_count)
    ]

    rng = np.random.default_rng(20260918)

    # ------------------------------ 预热 ------------------------------
    fill_batch(host_inputs[0], rng)
    cuda.memcpy_htod_async(device_inputs[0], host_inputs[0], streams[0])
    preprocess_batch(
        device_inputs[0],
        device_outputs[0],
        np.int32(batch_size),
        np.int32(height),
        np.int32(width),
        np.int32(crop_size),
        mean,
        inv_std,
        block=block,
        grid=grid,
        stream=streams[0],
    )
    streams[0].synchronize()

    # --------------------------- 流水线吞吐测试 ---------------------------
    num_batches = 100
    wall_start = time.perf_counter()

    for batch_id in range(num_batches):
        slot = batch_id % buffer_count
        stream = streams[slot]

        # 复用 slot 前，等待它上一次承担的任务完成。
        # 另一个 slot 仍可在 GPU 上工作。
        stream.synchronize()

        # 模拟准备当前 Batch；真实系统可替换为读取和解码。
        fill_batch(host_inputs[slot], rng)

        # 在同一 Stream 中先传输、再计算，数据依赖顺序由 Stream 保证。
        cuda.memcpy_htod_async(
            device_inputs[slot],
            host_inputs[slot],
            stream,
        )
        preprocess_batch(
            device_inputs[slot],
            device_outputs[slot],
            np.int32(batch_size),
            np.int32(height),
            np.int32(width),
            np.int32(crop_size),
            mean,
            inv_std,
            block=block,
            grid=grid,
            stream=stream,
        )

    # 等待所有仍在飞行中的 Batch 完成，端到端计时才闭合。
    for stream in streams:
        stream.synchronize()

    wall_seconds = time.perf_counter() - wall_start
    images_per_second = num_batches * batch_size / wall_seconds

    print(f"模拟端到端耗时: {wall_seconds:.6f} 秒")
    print(f"稳态吞吐量: {images_per_second:.2f} images/s")

    # --------------------------- 独立正确性校验 ---------------------------
    # 校验不混入上面的吞吐计时，也不要求每个生产 Batch 都回传 CPU。
    check_input = host_inputs[0]
    fill_batch(check_input, rng)
    expected = cpu_reference(check_input, crop_size, mean, stddev)
    actual = np.empty(output_shape, dtype=np.float32)

    cuda.memcpy_htod(device_inputs[0], check_input)
    preprocess_batch(
        device_inputs[0],
        device_outputs[0],
        np.int32(batch_size),
        np.int32(height),
        np.int32(width),
        np.int32(crop_size),
        mean,
        inv_std,
        block=block,
        grid=grid,
    )
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(actual, device_outputs[0])

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    print("正确性校验通过：GPU 结果与 NumPy 参考结果一致。")


if __name__ == "__main__":
    main()
