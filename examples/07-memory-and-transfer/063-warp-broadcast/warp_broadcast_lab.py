"""一组线程读同一个数，GPU 有什么捷径？（第7章 · 访存与传输）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：Numba、NumPy。
"""
import numpy as np
from numba import cuda, float32

THREADS = 256
TAPS = 5
WEIGHTS = np.array([0.40, 0.25, 0.20, 0.10, 0.05], dtype=np.float32)


@cuda.jit
def filter_global(x, weights, y):
    i = cuda.grid(1)
    if i < x.size:
        acc = float32(0.0)
        for k in range(TAPS):
            pos = i - k
            if pos >= 0:
                acc += weights[k] * x[pos]
        y[i] = acc


@cuda.jit
def filter_constant(x, y):
    weights = cuda.const.array_like(WEIGHTS)
    i = cuda.grid(1)
    if i < x.size:
        acc = float32(0.0)
        for k in range(TAPS):
            pos = i - k
            if pos >= 0:
                acc += weights[k] * x[pos]
        y[i] = acc

from importlib.metadata import version
from statistics import median

TABLE_SIZE = 256
TABLE = np.random.default_rng(7).uniform(0.1, 1.0, TABLE_SIZE).astype(np.float32)


@cuda.jit
def lookup_global(x, table, y, u):
    i = cuda.grid(1)
    if i < x.size:
        # 37 只是让不同 warp 改变查表起点，不是硬件参数。
        index = ((i // 32) * 37 + (i % 32) % u) % TABLE_SIZE
        y[i] = x[i] * table[index]


@cuda.jit
def lookup_constant(x, y, u):
    table = cuda.const.array_like(TABLE)
    i = cuda.grid(1)
    if i < x.size:
        index = ((i // 32) * 37 + (i % 32) % u) % TABLE_SIZE
        y[i] = x[i] * table[index]


def measure(kernel, blocks, args):
    for _ in range(3):
        kernel[blocks, THREADS](*args)
    cuda.synchronize()
    start, end = cuda.event(timing=True), cuda.event(timing=True)
    times = []
    for _ in range(5):
        start.record()
        for _ in range(20):
            kernel[blocks, THREADS](*args)
        end.record()
        end.synchronize()
        times.append(cuda.event_elapsed_time(start, end) / 20)
    return median(times), min(times), max(times)


def check_pair(label, pair, blocks, expected, out, benchmark):
    for name, kernel, args in pair:
        kernel[blocks, THREADS](*args)
        cuda.synchronize()
        got = out.copy_to_host()
        # 容差针对本脚本的 FP32 输入；不是任意算法的误差界。
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
        print(f"  {label}/{name}: correctness PASS")
    if benchmark:
        for name, kernel, args in pair:
            mid, low, high = measure(kernel, blocks, args)
            print(
                f"  {label}/{name}: median={mid:.6f} ms"
                f", group means range=[{low:.6f}, {high:.6f}] ms"
            )


def run_case(n, benchmark=False):
    host = np.random.default_rng(42).random(n, dtype=np.float32)
    x = cuda.to_device(host)
    y = cuda.device_array(n, dtype=np.float32)
    dw, dt = cuda.to_device(WEIGHTS), cuda.to_device(TABLE)
    blocks = (n + THREADS - 1) // THREADS
    print(f"N={n}")

    expected = np.convolve(
        host.astype(np.float64), WEIGHTS.astype(np.float64), mode="full"
    )[:n]
    pair = [
        ("global", filter_global, (x, dw, y)),
        ("constant", filter_constant, (x, y)),
    ]
    check_pair("filter", pair, blocks, expected, y, benchmark)

    positions = np.arange(n, dtype=np.int64)
    for u in (1, 4, 32):
        indices = ((positions // 32) * 37 + (positions % 32) % u) % TABLE_SIZE
        expected = host * TABLE[indices]
        pair = [
            ("global", lookup_global, (x, dt, y, u)),
            ("constant", lookup_constant, (x, y, u)),
        ]
        check_pair(f"lookup/u={u}", pair, blocks, expected, y, benchmark)


def main():
    if not cuda.is_available():
        raise SystemExit("需要可用的 NVIDIA CUDA 环境。")
    print("NumPy:", np.__version__)
    print("Numba:", version("numba"))
    print("Numba-CUDA:", version("numba-cuda"))
    gpu = cuda.get_current_device()
    print("GPU:", gpu.name, "compute capability:", gpu.compute_capability)
    print("FP32, block=256, warmup=3, repeats=20, samples=5")
    for n in (17, 257):
        run_case(n)
    for n in (1 << 18, 1 << 20):
        run_case(n, benchmark=True)


if __name__ == "__main__":
    main()
