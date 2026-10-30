"""CT强度映射教学实验：CPU检查可单独运行，GPU实验需要CuPy和CUDA。"""
import argparse
import platform
import time
import numpy as np

LO, SPAN = np.float32(-57), np.float32(221)


def prepare(x):
    # 在计时外统一为float32连续内存；只接受非空、有限的实数数据。
    x = np.asarray(x)
    # 必须在dtype转换前拒绝复数，否则float32转换会丢弃虚部后静默通过。
    if np.iscomplexobj(x):
        raise ValueError("输入必须为实数，不能包含复数。")
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("输入必须非空且不含NaN/Inf。")
    return x


def reference(x):
    # 与选定模型配置相同的数学映射；不冒充MONAI完整空间变换。
    return np.clip((x - LO) / SPAN, np.float32(0), np.float32(1))


def cases():
    edge = np.array([-1000, -57, -56.999, 0, 163.999, 164, 1000], np.float32)
    rng = np.random.default_rng(2026)
    return [edge] + [rng.uniform(-1000, 1000, n).astype(np.float32)
                     for n in (1, 255, 256, 257)] + [
        rng.uniform(-1000, 1000, (5, 7, 9)).astype(np.float32),
        np.arange(120, dtype=np.float32).reshape(4, 5, 6).transpose(2, 0, 1),
    ]


def cpu_check():
    expected = np.array([0, 0, 57 / 221, 1, 1], np.float32)
    np.testing.assert_allclose(reference(prepare([-1000, -57, 0, 164, 1000])),
                               expected, rtol=1e-6, atol=1e-6)
    for x in cases():
        x = prepare(x)
        y = reference(x)
        assert y.shape == x.shape and y.dtype == np.float32
        assert ((y >= 0) & (y <= 1)).all() and x.flags.c_contiguous
    for bad in ([], [np.nan], [np.inf], [-np.inf],
                [1 + 2j], np.array([1 + 0j, 2 + 0j])):
        try:
            prepare(bad)
        except ValueError:
            continue
        raise AssertionError("无效输入没有被拒绝。")
    # 只在CPU上检查索引分配，不能代替GPU越界检查。
    for n in (1, 255, 256, 257, 315):
        ids = [b * 256 + t for b in range((n + 255) // 256)
               for t in range(256) if b * 256 + t < n]
        assert ids == list(range(n))
    print("CPU检查通过：数学参考、布局转换、空/非有限/复数输入拒绝和索引分配；未验证CUDA。")


CUDA_SOURCE = r'''
extern "C" __global__
void normalize(const float* x, float* y, unsigned long long n,
               float lo, float span) {
    // 先提升类型再乘，避免用32位乘法计算大数组偏移。
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;  // 向上取整启动时，末尾多出的线程不读写内存。
    float t = (x[i] - lo) / span;  // 本体素的中间结果保留在线程内。
    y[i] = fminf(fmaxf(t, 0.0f), 1.0f);  // 写出范围为[0,1]的值。
}
'''


def gpu_experiment(shape, repeats):
    import cupy as cp
    from cupyx.profiler import benchmark

    raw = cp.RawKernel(CUDA_SOURCE, "normalize", options=("--std=c++11",))
    elem = cp.ElementwiseKernel(
        "float32 x, float32 lo, float32 span", "float32 y",
        "float t = (x - lo) / span; y = fminf(fmaxf(t, 0.0f), 1.0f);",
        "normalize_elementwise")  # 融合对照：让CuPy管理线程索引。

    def operations(d):
        # 三种方案的输出和临时缓冲区都在计时外申请。
        a, b, out_three, out_raw, out_elem = [cp.empty_like(d) for _ in range(5)]
        blocks = (d.size + 255) // 256

        def three():
            cp.subtract(d, LO, out=a)
            cp.divide(a, SPAN, out=b)
            cp.clip(b, np.float32(0), np.float32(1), out=out_three)

        def fused_raw():
            raw((blocks,), (256,), (d, out_raw, np.uint64(d.size), LO, SPAN))

        def fused_elementwise():
            elem(d, LO, SPAN, out_elem)

        return [("cupy_three_pass", three, out_three),
                ("cupy_elementwise", fused_elementwise, out_elem),
                ("cuda_raw", fused_raw, out_raw)]

    # 先验证边界和多种布局；GPU结果回传后与CPU参考比较。
    for original in cases():
        x = prepare(original)
        for name, fn, out in operations(cp.asarray(x)):
            fn()
            np.testing.assert_allclose(cp.asnumpy(out), reference(x),
                                       rtol=1e-6, atol=1e-6, err_msg=name)
    print("GPU合成数据检查通过；尚未代表真实CT或完整模型验证。")
    cp.show_config()  # 留下软件、运行时和设备信息，便于复现实验。
    print("Python", platform.python_version(), "NumPy", np.__version__)
    print("driver", cp.cuda.runtime.driverGetVersion())
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    print("GPU", props["name"], "shape", shape, "dtype float32", "block 256")

    rng = np.random.default_rng(2026)
    x = prepare(rng.uniform(-1000, 1000, shape))  # 合成强度，不模拟真实解剖。
    d = cp.asarray(x)
    expected = reference(x)
    for name, fn, out in operations(d):
        fn()  # 首次编译和执行发生在计时前。
        got = cp.asnumpy(out)
        max_abs = float(np.max(np.abs(got - expected)))
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
        # 不只报告"通过"，同时留下形状、dtype与最大绝对误差供复核。
        print(name, "shape", got.shape, "dtype", got.dtype,
              "max_abs_error", f"{max_abs:.3e}", "within_tolerance", True)
        result = benchmark(fn, (), n_warmup=10, n_repeat=repeats)
        us = result.gpu_times[0] * 1e6  # 当前设备上的CUDA事件时间，单位微秒。
        print(name, "GPU us median/p10/p90", np.percentile(us, [50, 10, 90]))
        if name == "cuda_raw":
            # 这是按一次读、一次写估算的有效带宽，不是DRAM硬件计数。
            print("raw effective GB/s", (8 * x.size) / (np.median(us) * 1e3))

    # CPU基线同样复用缓冲区；主机计时不与上述GPU事件冒充相同范围。
    a, b, y = [np.empty_like(x) for _ in range(3)]
    def cpu_three():
        np.subtract(x, LO, out=a)
        np.divide(a, SPAN, out=b)
        np.clip(b, np.float32(0), np.float32(1), out=y)
    for _ in range(10):
        cpu_three()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        cpu_three()
        times.append((time.perf_counter() - start) * 1e6)
    print("numpy CPU us median/p10/p90", np.percentile(times, [50, 10, 90]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--shape", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if min(args.shape) < 1 or args.repeats < 1:
        parser.error("shape和repeats必须为正整数。")
    cpu_check()
    if not args.cpu_only:
        gpu_experiment(tuple(args.shape), args.repeats)
