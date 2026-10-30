"""GPU 算得快，数据怎么搬也很关键：用 PyCUDA 拆解矩阵乘法的内存流（附完整代码）（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import time

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


# BLOCK_SIZE 会在 Python 端替换成编译期常量。
KERNEL_CODE = r"""
#define BLOCK_SIZE %(BLOCK_SIZE)s

__global__ void matrix_multiply_shared(
    const float *A,
    const float *B,
    float *C,
    int N
) {
    // 每个线程块协作缓存 A、B 的一个 tile。
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // block 选择 C 中的输出 tile。
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 当前线程在线程块内的位置。
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 当前线程负责输出 C[row, col]。
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // sum 在 K 维度的多轮循环中持续累加。
    float sum = 0.0f;

    // 沿 A 的列 / B 的行方向分块，也就是矩阵乘法的 K 维度。
    int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int tile_k_start = tile * BLOCK_SIZE;

        // 当前线程加载 A[row, tile_k_start + tx]。
        int a_col = tile_k_start + tx;
        if (row < N && a_col < N) {
            As[ty][tx] = A[row * N + a_col];
        } else {
            // 最后一个 tile 可能越过矩阵边界，越界元素补 0。
            As[ty][tx] = 0.0f;
        }

        // 当前线程加载 B[tile_k_start + ty, col]。
        int b_row = tile_k_start + ty;
        if (b_row < N && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            // 越界位置补 0，使不完整 tile 也能安全计算。
            Bs[ty][tx] = 0.0f;
        }

        // 等待线程块内的加载完成，保证共享内存数据可读。
        __syncthreads();

        // 使用本轮 tile 计算一段点积，并加到当前线程的 sum。
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 确保所有线程读完当前 tile，再允许下一轮覆盖共享内存。
        __syncthreads();
    }

    // 只写回合法的 C[row, col]。
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
"""


def average_seconds(elapsed, repeat_count):
    """把多次运行的总秒数换算成平均秒数。"""
    return elapsed / repeat_count


def main():
    # 矩阵维度；本示例实现 N×N 方阵。
    N = 1024

    # 一个线程块计算 C 的 16×16 区域。
    BLOCK_SIZE = 16

    # 分别测量传输和 Kernel 时的重复次数。
    TRANSFER_REPEATS = 10
    KERNEL_REPEATS = 20
    END_TO_END_REPEATS = 5

    if N <= 0 or BLOCK_SIZE <= 0:
        raise ValueError("N 和 BLOCK_SIZE 必须为正整数。")

    # 生成 float32 输入，与 CUDA Kernel 中的 float 类型一致。
    rng = np.random.default_rng(seed=42)
    A = rng.standard_normal((N, N)).astype(np.float32)
    B = rng.standard_normal((N, N)).astype(np.float32)

    # 把 BLOCK_SIZE 注入 CUDA 源码并编译 Kernel。
    source = KERNEL_CODE % {"BLOCK_SIZE": BLOCK_SIZE}
    module = SourceModule(source, options=["-O3"])
    matmul = module.get_function("matrix_multiply_shared")

    # 在 GPU 上为 A、B、C 分别申请显存。
    # 分配放在计时区间外，避免把分配开销混入传输或 Kernel 时间。
    A_gpu = cuda.mem_alloc(A.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    C_gpu = cuda.mem_alloc(A.nbytes)

    # 一个 block 负责 C 的一个 BLOCK_SIZE×BLOCK_SIZE 输出区域。
    block = (BLOCK_SIZE, BLOCK_SIZE, 1)
    grid = (
        (N + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) // BLOCK_SIZE,
        1,
    )

    # H2D：测量把 A、B 从 CPU 复制到 GPU 的主机端耗时。
    cuda.Context.synchronize()
    start = time.perf_counter()
    for _ in range(TRANSFER_REPEATS):
        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)
    h2d_total = time.perf_counter() - start
    h2d_average = average_seconds(h2d_total, TRANSFER_REPEATS)

    # 预热：首次启动及相关初始化不进入正式 Kernel 计时。
    matmul(A_gpu, B_gpu, C_gpu, np.int32(N), block=block, grid=grid)
    cuda.Context.synchronize()

    # 使用 CUDA Event 测量 GPU 时间线上的 Kernel 执行时间。
    # 这里不包括 H2D、D2H 和显存分配。
    start_event = cuda.Event()
    end_event = cuda.Event()

    start_event.record()
    for _ in range(KERNEL_REPEATS):
        matmul(A_gpu, B_gpu, C_gpu, np.int32(N), block=block, grid=grid)
    end_event.record()
    end_event.synchronize()

    # Event 时间以毫秒返回，除以 1000 转成秒。
    kernel_total = start_event.time_till(end_event) / 1000.0
    kernel_average = average_seconds(kernel_total, KERNEL_REPEATS)

    # 创建 CPU 输出数组，用于接收 GPU 上的 C。
    C_gpu_result = np.empty_like(A)

    # D2H：测量把 C 从 GPU 复制回 CPU 的主机端耗时。
    cuda.Context.synchronize()
    start = time.perf_counter()
    for _ in range(TRANSFER_REPEATS):
        cuda.memcpy_dtoh(C_gpu_result, C_gpu)
    d2h_total = time.perf_counter() - start
    d2h_average = average_seconds(d2h_total, TRANSFER_REPEATS)

    # 完整流程：计入输入拷贝、Kernel、输出拷贝。
    # 设备缓冲区仍在计时区间外申请并复用。
    # 同步 D2H 拷贝会等待此前的 Kernel 完成。
    cuda.Context.synchronize()
    start = time.perf_counter()
    for _ in range(END_TO_END_REPEATS):
        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)
        matmul(A_gpu, B_gpu, C_gpu, np.int32(N), block=block, grid=grid)
        cuda.memcpy_dtoh(C_gpu_result, C_gpu)
    flow_total = time.perf_counter() - start
    flow_average = average_seconds(flow_total, END_TO_END_REPEATS)

    # NumPy 矩阵乘法作为正确性参考。
    C_numpy = A @ B
    is_close = np.allclose(
        C_gpu_result,
        C_numpy,
        rtol=1e-3,
        atol=1e-3,
    )
    max_abs_error = np.max(np.abs(C_gpu_result - C_numpy))

    # 输出本机实测值。不同设备、驱动、系统负载下耗时都会变化。
    print(f"矩阵规模：{N}×{N}")
    print(f"线程块：{BLOCK_SIZE}×{BLOCK_SIZE}")
    print(f"H2D 平均耗时（A 和 B）：{h2d_average:.6f} 秒")
    print(f"Kernel 平均耗时：{kernel_average:.6f} 秒")
    print(f"D2H 平均耗时（C）：{d2h_average:.6f} 秒")
    print(f"H2D + Kernel + D2H 流程平均耗时：{flow_average:.6f} 秒")
    print(f"结果校验：{'通过' if is_close else '失败'}")
    print(f"最大绝对误差：{max_abs_error:.6e}")

    if not is_close:
        raise AssertionError("CUDA 计算结果与 NumPy 参考结果不一致。")


if __name__ == "__main__":
    main()
