"""GPU 算了 1000 次，温度为什么只前进了一步？（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy、Matplotlib。
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pycuda.autoinit  # 自动创建CUDA上下文
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


# ============================================================
# 1. CUDA内核：一维热传导显式有限差分
# ============================================================
KERNEL_CODE = r"""
__global__ void heat_step(
    const float *current,
    float *next,
    int n,
    float r)
{
    // 当前CUDA线程负责的全局网格点
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 向上取整的grid可能产生多余线程，必须先做越界保护
    if (idx >= n) {
        return;
    }

    // Dirichlet边界：两端始终保持环境相对温度0
    if (idx == 0 || idx == n - 1) {
        next[idx] = 0.0f;
        return;
    }

    // FTCS更新：读取第n时刻，写入第n+1时刻
    next[idx] = current[idx]
              + r * (current[idx - 1]
                   - 2.0f * current[idx]
                   + current[idx + 1]);
}
"""


# JIT编译发生在计时区域之外，避免把首次编译成本算进迭代时间
module = SourceModule(KERNEL_CODE, options=["-O3"])
heat_step = module.get_function("heat_step")


# ============================================================
# 2. 初始温度：中央出现一段局部热点
# ============================================================
def make_initial_temperature(n: int) -> np.ndarray:
    temperature = np.zeros(n, dtype=np.float32)

    center = n // 2
    half_width = max(32, n // 4096)
    temperature[center - half_width:center + half_width] = 1.0

    # 与内核边界条件保持一致
    temperature[0] = 0.0
    temperature[-1] = 0.0
    return temperature


# ============================================================
# 3. CPU参考实现：用于正确性验证，不假装是最优CPU版本
# ============================================================
def solve_cpu(initial: np.ndarray, r: float, timesteps: int):
    current = initial.copy()
    next_temperature = np.empty_like(current)

    start = time.perf_counter()

    for _ in range(timesteps):
        next_temperature[0] = 0.0
        next_temperature[-1] = 0.0

        next_temperature[1:-1] = (
            current[1:-1]
            + r
            * (
                current[:-2]
                - 2.0 * current[1:-1]
                + current[2:]
            )
        )

        # CPU端同样采用双缓冲
        current, next_temperature = next_temperature, current

    elapsed_s = time.perf_counter() - start
    return current, elapsed_s


# ============================================================
# 4. GPU预热：排除首次运行的一次性开销干扰
# ============================================================
def warm_up_gpu(r: float, block_size: int):
    warm_host = np.zeros(block_size, dtype=np.float32)
    warm_current = cuda.to_device(warm_host)
    warm_next = cuda.mem_alloc(warm_host.nbytes)

    heat_step(
        warm_current,
        warm_next,
        np.int32(warm_host.size),
        np.float32(r),
        block=(block_size, 1, 1),
        grid=(1, 1, 1),
    )
    cuda.Context.synchronize()


# ============================================================
# 5. GPU实现：显存驻留、双缓冲、一次回传
# ============================================================
def solve_gpu(
    initial: np.ndarray,
    r: float,
    timesteps: int,
    block_size: int = 256,
):
    n = initial.size
    grid_size = (n + block_size - 1) // block_size

    # 端到端计时：包含显存分配、H2D、迭代和D2H，不包含JIT编译
    e2e_start = time.perf_counter()

    current_gpu = cuda.to_device(initial)
    next_gpu = cuda.mem_alloc(initial.nbytes)

    start_event = cuda.Event()
    end_event = cuda.Event()

    start_event.record()

    for _ in range(timesteps):
        heat_step(
            current_gpu,
            next_gpu,
            np.int32(n),
            np.float32(r),
            block=(block_size, 1, 1),
            grid=(grid_size, 1, 1),
        )

        # 核心：本轮输出成为下一轮输入
        current_gpu, next_gpu = next_gpu, current_gpu

    end_event.record()
    end_event.synchronize()

    # PyCUDA Event返回毫秒
    kernel_ms = start_event.time_till(end_event)

    result = np.empty_like(initial)
    cuda.memcpy_dtoh(result, current_gpu)

    e2e_s = time.perf_counter() - e2e_start
    return result, kernel_ms / 1000.0, e2e_s


def main():
    # N可逐步改为2^10、2^12、...、2^20，观察规模拐点
    n = 1 << 16
    timesteps = 2000
    r = 0.4
    block_size = 256

    if not (0.0 < r <= 0.5):
        raise ValueError(
            f"显式一维热传导格式要求0 < r <= 0.5，当前r={r}"
        )

    initial = make_initial_temperature(n)

    warm_up_gpu(r, block_size)

    cpu_result, cpu_s = solve_cpu(initial, r, timesteps)
    gpu_result, kernel_s, e2e_s = solve_gpu(
        initial,
        r,
        timesteps,
        block_size,
    )

    # 正确性优先：先比较结果，再谈速度
    np.testing.assert_allclose(
        gpu_result,
        cpu_result,
        rtol=1e-5,
        atol=1e-5,
    )

    if not np.all(np.isfinite(gpu_result)):
        raise RuntimeError("结果中出现NaN或Inf")

    # 稳定且无热源时，不应无缘无故超过初始极值
    tolerance = 1e-5
    if gpu_result.min() < -tolerance or gpu_result.max() > 1.0 + tolerance:
        raise RuntimeError("结果违反预期的温度范围，请检查稳定性或边界条件")

    max_abs_error = np.max(np.abs(cpu_result - gpu_result))

    print(f"网格点数              : {n:,}")
    print(f"时间步数              : {timesteps:,}")
    print(f"扩散参数r             : {r}")
    print(f"CPU参考实现            : {cpu_s:.6f} s")
    print(f"GPU Kernel             : {kernel_s:.6f} s")
    print(f"GPU端到端              : {e2e_s:.6f} s")
    print(f"CPU/GPU端到端加速比    : {cpu_s / e2e_s:.2f}x")
    print(f"CPU/GPU最大绝对误差    : {max_abs_error:.8e}")

    # 可视化归一化位置与相对环境温度
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)

    plt.figure(figsize=(10, 5))
    plt.plot(x, initial, label="初始热点", linewidth=1.6)
    plt.plot(x, gpu_result, label="扩散后", linewidth=2.0)
    plt.xlabel("归一化位置 x")
    plt.ylabel("相对环境温度")
    plt.title("一维热传导：局部热点随时间扩散")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
