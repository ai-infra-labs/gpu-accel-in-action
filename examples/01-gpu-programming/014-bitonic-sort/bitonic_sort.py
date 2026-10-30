"""1024 个数，为什么这份朴素 CUDA 实现要启动 55 次 Kernel？（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


KERNEL_CODE = r"""
__global__ void bitonic_sort_step(
    float* data,
    int j,
    int k,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    // XOR 翻转 j 对应的索引位，得到本阶段比较伙伴。
    const int ixj = i ^ j;

    // 一对元素只由索引较小的一侧处理。
    if (ixj <= i || ixj >= n) {
        return;
    }

    // k 决定当前区域构建升序段还是降序段。
    const bool ascending = ((i & k) == 0);

    const float a = data[i];
    const float b = data[ixj];
    const bool should_swap = ascending ? (a > b) : (a < b);

    if (should_swap) {
        data[i] = b;
        data[ixj] = a;
    }
}
"""


MODULE = SourceModule(KERNEL_CODE, options=["-O3"])
BITONIC_STEP = MODULE.get_function("bitonic_sort_step")


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def validate_input(data: np.ndarray) -> None:
    if data.ndim != 1:
        raise ValueError("input must be a 1D array")
    if data.dtype != np.float32:
        raise ValueError("input dtype must be float32")
    if not is_power_of_two(data.size):
        raise ValueError("input length must be a power of two")
    if not np.all(np.isfinite(data)):
        raise ValueError("this demo requires finite values; NaN/Inf are rejected")


def stage_count(n: int) -> int:
    if not is_power_of_two(n):
        raise ValueError("n must be a power of two")

    p = n.bit_length() - 1
    return p * (p + 1) // 2


def launch_bitonic_sort(
    data_gpu,
    n: int,
    stream: cuda.Stream,
    block_size: int = 256,
) -> None:
    if not is_power_of_two(n):
        raise ValueError("n must be a power of two")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    # 设备上限校验：超过 MAX_THREADS_PER_BLOCK 的启动会直接失败。
    device = cuda.Context.get_device()
    max_threads = device.get_attribute(
        cuda.device_attribute.MAX_THREADS_PER_BLOCK
    )
    if block_size > max_threads:
        raise ValueError(f"block_size must be <= {max_threads}")

    grid_size = (n + block_size - 1) // block_size

    k = 2
    while k <= n:
        j = k // 2
        while j > 0:
            BITONIC_STEP(
                data_gpu,
                np.int32(j),
                np.int32(k),
                np.int32(n),
                block=(block_size, 1, 1),
                grid=(grid_size, 1, 1),
                stream=stream,
            )
            j //= 2
        k *= 2


def validate_result(actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.array_equal(actual, expected):
        mismatch = np.flatnonzero(actual != expected)
        first = int(mismatch[0]) if mismatch.size else -1
        raise AssertionError(
            f"GPU result differs from CPU reference at index {first}"
        )

    if actual.size > 1 and not np.all(actual[:-1] <= actual[1:]):
        raise AssertionError("GPU result is not monotonic")


def benchmark_sort(
    original: np.ndarray,
    block_size: int = 256,
    warmup: int = 5,
    iterations: int = 50,
) -> dict:
    """只测 GPU 排序阶段；不包含 H2D、D2H 与显存分配。"""
    n = original.size
    stream = cuda.Stream()
    data_gpu = cuda.mem_alloc(original.nbytes)

    # 预热：复用同一块显存，每轮恢复输入。
    for _ in range(warmup):
        cuda.memcpy_htod(data_gpu, original)
        launch_bitonic_sort(data_gpu, n, stream, block_size)
        stream.synchronize()

    elapsed_ms = []

    # Event 可以复用：每轮已等待 end 到达，下一轮重新 record 即可。
    start = cuda.Event()
    end = cuda.Event()

    for _ in range(iterations):
        # 同步 H2D 在计时区间之外完成。
        cuda.memcpy_htod(data_gpu, original)

        start.record(stream)
        launch_bitonic_sort(data_gpu, n, stream, block_size)
        end.record(stream)
        end.synchronize()

        elapsed_ms.append(float(start.time_till(end)))

    samples = np.asarray(elapsed_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
    }


def run_tests(block_size: int = 256) -> int:
    """多规模、多输入形态的正确性回归；golden 仍是 CPU 侧 np.sort。"""
    stream = cuda.Stream()
    cases = {
        "random": lambda n, rng: rng.random(n, dtype=np.float32),
        "sorted": lambda n, rng: np.sort(rng.random(n, dtype=np.float32)),
        "reversed": lambda n, rng: np.sort(
            rng.random(n, dtype=np.float32)
        )[::-1].copy(),
        "duplicates": lambda n, rng: rng.integers(
            0, 3, size=n
        ).astype(np.float32),
        "all_equal": lambda n, rng: np.full(n, 1.5, dtype=np.float32),
    }

    checked = 0
    seed = 0
    for n in (1, 2, 4, 8, 32, 256, 1024):
        for make_case in cases.values():
            seed += 1
            host = np.ascontiguousarray(
                make_case(n, np.random.default_rng(seed)), dtype=np.float32
            )

            gpu = cuda.mem_alloc(host.nbytes)
            cuda.memcpy_htod(gpu, host)
            launch_bitonic_sort(gpu, n, stream, block_size)
            stream.synchronize()

            out = np.empty_like(host)
            cuda.memcpy_dtoh(out, gpu)
            gpu.free()

            validate_result(out, np.sort(host.copy()))
            checked += 1

    return checked


def main() -> None:
    n = 1024
    block_size = 256
    rng = np.random.default_rng(42)
    original = rng.random(n, dtype=np.float32)

    validate_input(original)
    expected = np.sort(original.copy())

    stream = cuda.Stream()
    data_gpu = cuda.mem_alloc(original.nbytes)
    cuda.memcpy_htod(data_gpu, original)

    launch_bitonic_sort(data_gpu, n, stream, block_size)
    stream.synchronize()

    actual = np.empty_like(original)
    cuda.memcpy_dtoh(actual, data_gpu)
    validate_result(actual, expected)

    print("Correctness: PASS")
    print("Input size:", n)
    print("Kernel stages:", stage_count(n))
    print("First 10 values:", actual[:10])

    checked = run_tests(block_size)
    print(f"Regression: PASS ({checked} cases)")

    stats = benchmark_sort(
        original,
        block_size=block_size,
        warmup=5,
        iterations=50,
    )

    print("Benchmark: kernel sequence only")
    print(f"P50: {stats['p50_ms']:.4f} ms")
    print(f"P95: {stats['p95_ms']:.4f} ms")
    print(f"Min: {stats['min_ms']:.4f} ms")
    print(f"Max: {stats['max_ms']:.4f} ms")


if __name__ == "__main__":
    main()
