"""把声音拆成频率，究竟发生了什么？（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import sys

import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


KERNEL_CODE = r"""
#include <math_constants.h>

__global__ void dft_real_to_complex_tiled(
    const float* __restrict__ input,
    float* __restrict__ real_output,
    float* __restrict__ imag_output,
    int N
) {
    // 一个线程负责一个输出频率桶 k。
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid_output = (k < N);

    float real_sum = 0.0f;
    float imag_sum = 0.0f;

    // Host 启动 Kernel 时传入 block_size * sizeof(float) 字节。
    extern __shared__ float input_tile[];

    // 每轮让整个 Block 协作搬运一个输入 Tile。
    for (int tile_start = 0;
         tile_start < N;
         tile_start += blockDim.x) {

        const int load_n = tile_start + threadIdx.x;
        input_tile[threadIdx.x] =
            (load_n < N) ? input[load_n] : 0.0f;

        // 所有线程必须到达：当前 Tile 才可安全读取。
        __syncthreads();

        if (valid_output) {
            const int remaining = N - tile_start;
            const int tile_count =
                remaining < blockDim.x ? remaining : blockDim.x;

            for (int local_n = 0; local_n < tile_count; ++local_n) {
                const int n = tile_start + local_n;
                const float angle =
                    -2.0f * CUDART_PI_F * (float)k * (float)n / (float)N;

                float sin_value;
                float cos_value;
                sincosf(angle, &sin_value, &cos_value);

                const float x = input_tile[local_n];
                real_sum += x * cos_value;
                imag_sum += x * sin_value;
            }
        }

        // 防止下一轮提前覆盖仍在被读取的 Tile。
        __syncthreads();
    }

    if (valid_output) {
        real_output[k] = real_sum;
        imag_output[k] = imag_sum;
    }
}
"""


MODULE = SourceModule(KERNEL_CODE, options=["-O3"])
DFT_KERNEL = MODULE.get_function("dft_real_to_complex_tiled")


def launch_dft(input_gpu, real_gpu, imag_gpu, n, block_size=256):
    if n <= 0:
        raise ValueError("n must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    grid_size = (n + block_size - 1) // block_size
    shared_bytes = block_size * np.dtype(np.float32).itemsize

    DFT_KERNEL(
        input_gpu,
        real_gpu,
        imag_gpu,
        np.int32(n),
        block=(block_size, 1, 1),
        grid=(grid_size, 1, 1),
        shared=shared_bytes,
    )


def gpu_dft(x, block_size=256):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array")

    n = x.size
    item_bytes = np.dtype(np.float32).itemsize

    input_gpu = cuda.to_device(x)
    real_gpu = cuda.mem_alloc(n * item_bytes)
    imag_gpu = cuda.mem_alloc(n * item_bytes)

    launch_dft(input_gpu, real_gpu, imag_gpu, n, block_size)
    cuda.Context.synchronize()

    real = np.empty(n, dtype=np.float32)
    imag = np.empty(n, dtype=np.float32)
    cuda.memcpy_dtoh(real, real_gpu)
    cuda.memcpy_dtoh(imag, imag_gpu)

    return real.astype(np.float64) + 1j * imag.astype(np.float64)


def validate_result(actual, expected, x, rtol=5e-4):
    """容差按输入幅度标定，而不是全局固定值。

    朴素 DFT 的 float32 误差近似正比于「输入 L1 范数 × eps × 累加层级数」，
    所以 atol 取 64 * eps * max(1, sum(abs(x)))。sum(abs(x)) 是唯一的规模因子：
    频谱峰值已经由输入决定，不应再用 max(abs(expected)) 把 N 重复放大一次——
    那会得到约 16*eps*N^2 的容差（N=1024 时约 2.0），足以放过显而易见的错误。
    系数 64 是工程安全系数，不是数学定理；更换规模、输入分布或数学库后应重新标定。
    rtol 覆盖接近零值附近的相对误差。除通过判定外，还返回最大绝对误差、
    相对 L2 误差与 atol，便于解释误差来源。
    """
    if actual.shape != expected.shape:
        raise AssertionError(
            f"shape mismatch: actual={actual.shape}, expected={expected.shape}"
        )

    eps = np.finfo(np.float32).eps
    input_l1 = float(np.sum(np.abs(x), dtype=np.float64))
    atol = 64.0 * eps * max(1.0, input_l1)

    abs_error = np.abs(actual - expected)
    denom = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    relative_l2 = float(np.linalg.norm(actual - expected) / denom)

    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        index = int(np.argmax(abs_error))
        raise AssertionError(
            "DFT validation failed: "
            f"index={index}, max_abs_error={abs_error[index]:.6e}, "
            f"actual={actual[index]}, expected={expected[index]}, "
            f"atol={atol:.6e}"
        )

    return {
        "max_abs_error": float(np.max(abs_error)),
        "relative_l2_error": relative_l2,
        "atol": float(atol),
    }


def benchmark_kernel(x, block_size=256, warmup=5, iterations=50):
    """只测 Kernel，不包含 H2D、D2H 和显存分配。"""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.size
    item_bytes = np.dtype(np.float32).itemsize

    input_gpu = cuda.to_device(x)
    real_gpu = cuda.mem_alloc(n * item_bytes)
    imag_gpu = cuda.mem_alloc(n * item_bytes)

    for _ in range(warmup):
        launch_dft(input_gpu, real_gpu, imag_gpu, n, block_size)
    cuda.Context.synchronize()

    elapsed_ms = []
    start = cuda.Event()
    end = cuda.Event()

    for _ in range(iterations):
        start.record()
        launch_dft(input_gpu, real_gpu, imag_gpu, n, block_size)
        end.record()
        end.synchronize()
        elapsed_ms.append(float(start.time_till(end)))

    times = np.asarray(elapsed_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def self_test(sample_rate=8192.0, n_values=(64, 256, 1024, 4096)):
    """三类输入 × 四个规模，验证 Kernel 与参考实现的一致性。

    - random：随机输入，最容易暴露符号或公式错误；
    - on-bin-tones：落桶双音（对每个 N 取整数频率桶 f = k·fs/N）；
    - off-bin-windowed：443 Hz 非落桶音 + Hann 窗，覆盖频谱泄漏场景。
    """
    print("self-test (4 sizes × 3 cases, all must PASS):")
    for n in n_values:
        t = np.arange(n, dtype=np.float32) / np.float32(sample_rate)
        window = (
            0.5
            - 0.5
            * np.cos(
                2.0 * np.pi * np.arange(n, dtype=np.float32) / np.float32(n)
            )
        ).astype(np.float32)
        # 落桶双音：频率按 f = k * fs / N 选整数桶。
        # 固定 440 / 1000 Hz 只在 N=1024 / 4096 下落桶
        # （440*64/8192 = 3.4375，N=64 时并非整数桶）。
        bin1 = max(5, n // 16)    # 低频桶
        bin2 = max(11, n // 8)    # 高频桶（均低于 Nyquist 桶 n/2）
        f_on1 = bin1 * sample_rate / n
        f_on2 = bin2 * sample_rate / n
        cases = {
            "random": np.random.default_rng(0)
            .standard_normal(n)
            .astype(np.float32),
            "on-bin-tones": (
                np.sin(2.0 * np.pi * f_on1 * t)
                + 0.35 * np.sin(2.0 * np.pi * f_on2 * t + 0.4)
            ).astype(np.float32),
            "off-bin-windowed": (
                np.sin(2.0 * np.pi * 443.0 * t) * window
            ).astype(np.float32),
        }
        for name, x in cases.items():
            actual = gpu_dft(x)
            expected = np.fft.fft(x)
            metrics = validate_result(actual, expected, x)
            print(
                f"  N={n:<5} {name:<18} PASS  "
                f"max_abs={metrics['max_abs_error']:.3e}  "
                f"rel_l2={metrics['relative_l2_error']:.3e}  "
                f"atol={metrics['atol']:.3e}"
            )
    # 任何用例不通过都会在 validate_result 内抛 AssertionError，
    # 走不到打印与返回；此处不需要额外的通过标志。
    return True


def main():
    n = 1024
    sample_rate = 8192.0
    block_size = 256

    # 打印运行环境，便于复现（对应第 11 节的记录项）。
    device = cuda.Device(0)
    print(f"GPU: {device.name()}  Compute Capability: {device.compute_capability()}")
    print(f"PyCUDA: {pycuda.VERSION_TEXT}, NumPy: {np.__version__}")

    # 构造 440 Hz 与 1000 Hz 的双音信号。
    t = np.arange(n, dtype=np.float32) / np.float32(sample_rate)
    x = (
        1.0 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.35 * np.sin(2.0 * np.pi * 1000.0 * t + 0.4)
    ).astype(np.float32)

    actual = gpu_dft(x, block_size=block_size)
    expected = np.fft.fft(x)
    metrics = validate_result(actual, expected, x)

    # 只查看非负频率一侧；幅度采用 2/N 标定。
    positive_count = n // 2 + 1
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    amplitude = 2.0 * np.abs(actual[:positive_count]) / n
    amplitude[0] *= 0.5
    if n % 2 == 0:
        amplitude[-1] *= 0.5

    peak_indices = np.argsort(amplitude)[-5:][::-1]

    print("Correctness: PASS")
    print(
        f"max_abs_error={metrics['max_abs_error']:.3e}, "
        f"relative_l2_error={metrics['relative_l2_error']:.3e}, "
        f"atol={metrics['atol']:.3e}"
    )
    print(f"frequency bin spacing: {sample_rate / n:.3f} Hz")
    print("top positive-frequency bins:")
    for index in peak_indices:
        print(f"  {freqs[index]:8.2f} Hz  amplitude={amplitude[index]:.6f}")

    stats = benchmark_kernel(
        x,
        block_size=block_size,
        warmup=5,
        iterations=50,
    )
    print("kernel-only benchmark:", stats)


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        self_test()
        sys.exit(0)
    main()
