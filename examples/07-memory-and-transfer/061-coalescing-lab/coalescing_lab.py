"""同样搬一张表，GPU 为什么还要看访问方式？（第7章 · 访存与传输）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：Numba、NumPy。
"""
from importlib.metadata import version
from statistics import median
import numpy as np
from numba import cuda

BLOCK = (32, 8)
SHAPES = [(512, 768), (2048, 3072), (4096, 6144)]


@cuda.jit
def copy_rows(a, c):
    x, y = cuda.grid(2)
    if y < a.shape[0] and x < a.shape[1]:
        c[y, x] = a[y, x]


@cuda.jit
def transpose_read_contiguous(a, b):
    # x 对应输入列：读取沿行，写入沿列。
    x, y = cuda.grid(2)
    if y < a.shape[0] and x < a.shape[1]:
        b[x, y] = a[y, x]


@cuda.jit
def transpose_write_contiguous(a, b):
    # x 对应输出列：写入沿行，读取沿列。
    x, y = cuda.grid(2)
    if y < b.shape[0] and x < b.shape[1]:
        b[y, x] = a[x, y]


def grid_for(rows, cols):
    return (
        (cols + BLOCK[0] - 1) // BLOCK[0],
        (rows + BLOCK[1] - 1) // BLOCK[1],
    )


def measure(kernel, grid, a, out, repeats=20, samples=7):
    for _ in range(5):
        kernel[grid, BLOCK](a, out)
    cuda.synchronize()
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    timings = []
    for _ in range(samples):
        start.record()
        for _ in range(repeats):
            kernel[grid, BLOCK](a, out)
        end.record()
        end.synchronize()
        timings.append(cuda.event_elapsed_time(start, end) / repeats)
    return median(timings), min(timings), max(timings)


def run_case(shape, benchmark=True):
    m, n = shape
    rng = np.random.default_rng(42)
    host = rng.random(shape, dtype=np.float32)
    a = cuda.to_device(host)
    copied = cuda.device_array((m, n), dtype=np.float32)
    transposed = cuda.device_array((n, m), dtype=np.float32)
    tasks = [
        ("copy", copy_rows, grid_for(m, n), copied, host),
        ("transpose/read", transpose_read_contiguous,
         grid_for(m, n), transposed, host.T),
        ("transpose/write", transpose_write_contiguous,
         grid_for(n, m), transposed, host.T),
    ]

    # 每个版本单独验证。复制没有浮点运算，可逐元素精确比较。
    for name, kernel, grid, out, expected in tasks:
        kernel[grid, BLOCK](a, out)
        cuda.synchronize()
        np.testing.assert_array_equal(out.copy_to_host(), expected)
    print(f"shape={shape}, correctness: PASS")

    if benchmark:
        useful_bytes = 2 * host.nbytes
        for name, kernel, grid, out, _ in tasks:
            mid_ms, low_ms, high_ms = measure(kernel, grid, a, out)
            if mid_ms <= 0:
                raise RuntimeError("计时分辨率不足，请增大问题规模或重复次数。")
            gbps = useful_bytes / (mid_ms / 1000) / 1e9
            print(
                f"  {name:17s} {mid_ms:.6f} ms, {gbps:.2f} GB/s"
                f"; group means range=[{low_ms:.6f}, {high_ms:.6f}] ms"
            )


def main():
    if not cuda.is_available():
        raise SystemExit("需要可用的 NVIDIA CUDA 环境。")
    print("NumPy:", np.__version__)
    print("Numba:", version("numba"))
    print("Numba-CUDA:", version("numba-cuda"))
    gpu = cuda.get_current_device()
    print("GPU:", gpu.name, "compute capability:", gpu.compute_capability)
    print("block:", BLOCK, "dtype: float32, C-contiguous")
    print("warmup=5, repeats=20, samples=7; default stream")
    # 非整除尺寸只检查正确性，不拿极小任务作带宽结论。
    for shape in [(17, 35), (37, 19)]:
        run_case(shape, benchmark=False)
    for shape in SHAPES:
        run_case(shape)


if __name__ == "__main__":
    main()
