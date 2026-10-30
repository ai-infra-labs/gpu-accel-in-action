"""共享内存这么快，为什么不能全放进去？（第7章 · 访存与传输）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：Numba、NumPy。
"""
import numpy as np
from numba import cuda, float32

TILE = 16  # 编译时常量；共享数组形状与启动块尺寸必须一致。


@cuda.jit
def matmul_naive(a, b, c):
    col, row = cuda.grid(2)
    if row < c.shape[0] and col < c.shape[1]:
        acc = float32(0.0)
        for k in range(a.shape[1]):
            acc += a[row, k] * b[k, col]
        c[row, col] = acc


@cuda.jit
def matmul_tiled(a, b, c):
    sa = cuda.shared.array((TILE, TILE), dtype=float32)
    sb = cuda.shared.array((TILE, TILE), dtype=float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE + tx
    row = cuda.blockIdx.y * TILE + ty
    acc = float32(0.0)

    for i in range((a.shape[1] + TILE - 1) // TILE):
        ak = i * TILE + tx  # A 的全局列号。
        bk = i * TILE + ty  # B 的全局行号。
        sa[ty, tx] = float32(0.0)
        sb[ty, tx] = float32(0.0)
        if row < a.shape[0] and ak < a.shape[1]:
            sa[ty, tx] = a[row, ak]
        if bk < b.shape[0] and col < b.shape[1]:
            sb[ty, tx] = b[bk, col]
        cuda.syncthreads()

        for j in range(TILE):
            acc += sa[ty, j] * sb[j, tx]
        cuda.syncthreads()

    if row < c.shape[0] and col < c.shape[1]:
        c[row, col] = acc

from importlib.metadata import version
from statistics import median


def measure(kernel, grid, block, a, b, c):
    for _ in range(3):
        kernel[grid, block](a, b, c)
    cuda.synchronize()
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    timings = []
    for _ in range(5):
        start.record()
        for _ in range(5):
            kernel[grid, block](a, b, c)
        end.record()
        end.synchronize()
        timings.append(cuda.event_elapsed_time(start, end) / 5)
    return median(timings), min(timings), max(timings)


def run_case(m, k, n, benchmark=False):
    rng = np.random.default_rng(42)
    # 小幅随机输入，避免本教学容差被不必要的大数值放大。
    ah = (rng.random((m, k), dtype=np.float32) - np.float32(0.5))
    bh = (rng.random((k, n), dtype=np.float32) - np.float32(0.5))
    # 用 CPU float64 结果作为数值参照，不将其耗时与 GPU 对比。
    expected = ah.astype(np.float64) @ bh.astype(np.float64)
    a, b = cuda.to_device(ah), cuda.to_device(bh)
    c = cuda.device_array((m, n), dtype=np.float32)
    block = (TILE, TILE)
    grid = ((n + TILE - 1) // TILE, (m + TILE - 1) // TILE)
    tasks = [("naive", matmul_naive), ("tiled", matmul_tiled)]
    print(f"shape: M={m}, K={k}, N={n}")
    for name, kernel in tasks:
        kernel[grid, block](a, b, c)
        cuda.synchronize()
        got = c.copy_to_host()
        # 容差用于这些 FP32 小幅随机输入，不是通用精度承诺。
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
        error = np.max(np.abs(got.astype(np.float64) - expected))
        print(f"  {name}: correctness PASS, max_abs_error={error:.3e}")
    if benchmark:
        for name, kernel in tasks:
            mid, low, high = measure(kernel, grid, block, a, b, c)
            print(
                f"  {name}: median={mid:.6f} ms"
                f", group means range=[{low:.6f}, {high:.6f}] ms"
            )


def main():
    if not cuda.is_available():
        raise SystemExit("需要可用的 NVIDIA CUDA 环境。")
    print("NumPy:", np.__version__)
    print("Numba:", version("numba"))
    print("Numba-CUDA:", version("numba-cuda"))
    gpu = cuda.get_current_device()
    print("GPU:", gpu.name, "compute capability:", gpu.compute_capability)
    print("TILE=16, FP32, warmup=3, repeats=5, samples=5")
    for shape in [(5, 23, 7), (19, 35, 21), (33, 7, 17)]:
        run_case(*shape)
    for shape in [(256, 256, 256), (512, 512, 512)]:
        run_case(*shape, benchmark=True)


if __name__ == "__main__":
    main()
