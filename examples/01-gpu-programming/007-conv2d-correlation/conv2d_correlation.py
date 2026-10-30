"""Kernel 能编译，不代表它在算卷积：拆解 PyCUDA 二维卷积的三个反转（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy、SciPy。
"""
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from scipy.signal import correlate2d

kernel_code = r"""
__global__ void correlation_2d(
    const float *input,      // 输入图像（只读）
    const float *kernel,     // 卷积核（只读）
    float *output,           // 输出图像（每个线程只写一个像素）
    int input_width,
    int input_height,
    int kernel_width,
    int kernel_height)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid 向上取整后会多出无效线程，直接返回
    if (tx >= input_width || ty >= input_height) {
        return;
    }

    // 教学版约定奇数核，half 即邻域半径
    int half_k_width  = kernel_width / 2;
    int half_k_height = kernel_height / 2;

    float value = 0.0f;

    // 直接遍历卷积核数组：循环次数 = 核尺寸，下标永不越界
    for (int ky = 0; ky < kernel_height; ++ky) {
        for (int kx = 0; kx < kernel_width; ++kx) {
            // 核数组坐标 → 相对坐标 → 输入图像绝对坐标
            int x = tx + kx - half_k_width;
            int y = ty + ky - half_k_height;

            // 越界位置等价于零填充，不参与累加
            if (x >= 0 && x < input_width &&
                y >= 0 && y < input_height) {
                value += input[y * input_width + x]
                       * kernel[ky * kernel_width + kx];
            }
        }
    }

    // 一个线程只写一个输出位置：无写冲突，不需要原子操作
    output[ty * input_width + tx] = value;
}
"""

# 固定随机种子，保证实验可重复
rng = np.random.default_rng(42)

input_width = 513           # 故意避开 16 的整数倍，验证 Grid 向上取整
input_height = 513
kernel_width = 3
kernel_height = 3

if kernel_width % 2 == 0 or kernel_height % 2 == 0:
    raise ValueError("当前示例只支持奇数尺寸卷积核")

input_data = rng.random((input_height, input_width), dtype=np.float32)
kernel = rng.random((kernel_height, kernel_width), dtype=np.float32)
output_data = np.empty_like(input_data)

# 编译并获取 Kernel
mod = SourceModule(kernel_code)
correlation_2d = mod.get_function("correlation_2d")

# Host → Device
input_gpu = cuda.to_device(input_data)
kernel_gpu = cuda.to_device(kernel)
output_gpu = cuda.mem_alloc(output_data.nbytes)

block_size = (16, 16, 1)
grid_size = (
    (input_width + block_size[0] - 1) // block_size[0],
    (input_height + block_size[1] - 1) // block_size[1],
    1,
)

# CUDA Event 计时：只包围 Kernel，不含传输与回拷
start_event = cuda.Event()
end_event = cuda.Event()
start_event.record()
correlation_2d(
    input_gpu,
    kernel_gpu,
    output_gpu,
    np.int32(input_width),
    np.int32(input_height),
    np.int32(kernel_width),
    np.int32(kernel_height),
    block=block_size,
    grid=grid_size,
)
end_event.record()
end_event.synchronize()
kernel_time_ms = start_event.time_till(end_event)

# Device → Host
cuda.memcpy_dtoh(output_data, output_gpu)

# 参考实现：互相关用 correlate2d（不翻转核），same + 零填充
output_cpu = correlate2d(
    input_data, kernel, mode="same", boundary="fill", fillvalue=0
).astype(np.float32)

# 全量断言校验
np.testing.assert_allclose(output_data, output_cpu, rtol=1e-5, atol=1e-5)

# 误差量级展示（发回填时一并记录，让读者看到的不只是 PASS）
max_abs_error = float(np.max(np.abs(output_data - output_cpu)))
mean_abs_error = float(np.mean(np.abs(output_data - output_cpu)))

print(f"Kernel 执行时间：{kernel_time_ms:.3f} ms")
print(f"最大绝对误差：{max_abs_error:.8e}")
print(f"平均绝对误差：{mean_abs_error:.8e}")
print("GPU 结果与 SciPy correlate2d 一致")
print(output_data[:5, :5])
