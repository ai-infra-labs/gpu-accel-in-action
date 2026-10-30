"""PyCUDA 手写 3D 卷积：26 万个输出线程，为什么同一批输入被反复访问？（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import time

import numpy as np
import pycuda.autoinit  # 自动初始化 CUDA 设备、上下文与清理流程
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


KERNEL_CODE = r"""
__global__ void conv3d_same(
    const float *__restrict__ input,
    const float *__restrict__ kernel,
    float *__restrict__ output,
    int width,
    int height,
    int depth,
    int kernel_size)
{
    // 当前线程对应的输出体素坐标
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Grid 向上取整后，边缘 Block 可能包含越界线程
    if (x >= width || y >= height || z >= depth) {
        return;
    }

    const int radius = kernel_size / 2;
    float value = 0.0f;

    // 遍历卷积核的深度、高度和宽度
    for (int kz = 0; kz < kernel_size; ++kz) {
        const int iz = z + kz - radius;

        for (int ky = 0; ky < kernel_size; ++ky) {
            const int iy = y + ky - radius;

            for (int kx = 0; kx < kernel_size; ++kx) {
                const int ix = x + kx - radius;

                // 越界位置视为 0，即零填充
                if (ix >= 0 && ix < width &&
                    iy >= 0 && iy < height &&
                    iz >= 0 && iz < depth) {

                    const int input_index =
                        (iz * height + iy) * width + ix;

                    const int kernel_index =
                        (kz * kernel_size + ky) * kernel_size + kx;

                    value += input[input_index] * kernel[kernel_index];
                }
            }
        }
    }

    const int output_index = (z * height + y) * width + x;
    output[output_index] = value;
}
"""


def generate_data(width, height, depth, rng):
    """生成 C-order、float32 三维体数据，形状为 [D, H, W]。"""
    data = rng.random((depth, height, width), dtype=np.float32)
    return np.ascontiguousarray(data)


def generate_kernel(kernel_size, rng):
    """生成正奇数尺寸的三维卷积核。"""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size 必须是正奇数")

    weights = rng.random(
        (kernel_size, kernel_size, kernel_size),
        dtype=np.float32,
    )

    # 归一化不是卷积的必要条件，只是让示例输出数值更平稳
    weights /= weights.sum(dtype=np.float32)
    return np.ascontiguousarray(weights)


def conv3d_reference(input_data, kernel):
    """NumPy 参考实现：零填充、步长 1、三维互相关。"""
    depth, height, width = input_data.shape
    kernel_size = kernel.shape[0]

    if kernel.shape != (kernel_size, kernel_size, kernel_size):
        raise ValueError("kernel 必须是立方体")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size 必须是奇数")

    radius = kernel_size // 2
    padded = np.pad(
        input_data,
        ((radius, radius), (radius, radius), (radius, radius)),
        mode="constant",
    )
    output = np.zeros_like(input_data)

    # 外层只遍历较小的卷积核，体数据计算交给 NumPy 切片
    for kz in range(kernel_size):
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                output += (
                    padded[
                        kz:kz + depth,
                        ky:ky + height,
                        kx:kx + width,
                    ]
                    * kernel[kz, ky, kx]
                )

    return output


def main():
    width, height, depth = 64, 64, 64
    kernel_size = 3
    repeats = 100
    rng = np.random.default_rng(42)

    input_data = generate_data(width, height, depth, rng)
    kernel = generate_kernel(kernel_size, rng)
    output_gpu = np.empty_like(input_data)

    # 编译 CUDA C Kernel 并获取函数句柄
    module = SourceModule(KERNEL_CODE)
    conv3d_kernel = module.get_function("conv3d_same")

    # H2D：输入和权重复制到 GPU；输出只分配，不需要预先复制零值
    input_device = cuda.to_device(input_data)
    kernel_device = cuda.to_device(kernel)
    output_device = cuda.mem_alloc(output_gpu.nbytes)

    # 256 线程作为起始实验点；最佳形状需要基准测试确定
    block_size = (8, 8, 4)
    grid_size = (
        (width + block_size[0] - 1) // block_size[0],
        (height + block_size[1] - 1) // block_size[1],
        (depth + block_size[2] - 1) // block_size[2],
    )

    args = (
        input_device,
        kernel_device,
        output_device,
        np.int32(width),
        np.int32(height),
        np.int32(depth),
        np.int32(kernel_size),
    )

    # 预热：避免首次调用开销污染正式计时
    conv3d_kernel(*args, block=block_size, grid=grid_size)
    cuda.Context.synchronize()

    # CUDA Event 测量 GPU 时间
    start_event = cuda.Event()
    end_event = cuda.Event()

    start_event.record()
    for _ in range(repeats):
        conv3d_kernel(*args, block=block_size, grid=grid_size)
    end_event.record()
    end_event.synchronize()

    average_kernel_ms = start_event.time_till(end_event) / repeats

    # D2H：DeviceAllocation 使用 memcpy_dtoh 复制回 NumPy 数组
    cuda.memcpy_dtoh(output_gpu, output_device)

    # CPU 参考结果与耗时
    cpu_start = time.perf_counter()
    output_cpu = conv3d_reference(input_data, kernel)
    cpu_ms = (time.perf_counter() - cpu_start) * 1000.0

    # 浮点并行计算不应使用逐位相等判断
    np.testing.assert_allclose(
        output_gpu,
        output_cpu,
        rtol=1e-5,
        atol=1e-6,
    )

    max_abs_error = np.max(np.abs(output_gpu - output_cpu))

    print(f"输入形状[D,H,W]：{input_data.shape}")
    print(f"Block：{block_size}，Grid：{grid_size}")
    print(f"平均 Kernel 时间：{average_kernel_ms:.4f} ms")
    print(f"NumPy 参考实现时间：{cpu_ms:.4f} ms")
    print(f"最大绝对误差：{max_abs_error:.8e}")
    print("输出左上前方的 2×3×3 子区域：")
    print(output_gpu[:2, :3, :3])


if __name__ == "__main__":
    main()
