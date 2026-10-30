"""附录 A：正确性套件与 Benchmark 工具。

这是 reduction_scan.py 的配套脚本，复用其中的 kernel 与启动函数。
"""

import argparse
import time

import numpy as np

import pycuda.autoinit  # 创建 CUDA Context
import pycuda.driver as cuda

from reduction_scan import (
    build_reduction_workspace,
    build_scan_plan,
    ceil_div,
    launch_global_reduction,
    launch_global_scan,
    validate_config,
)

# 教学校验参数：只对本文示例负载有意义，不是通用误差界。
ERROR_TOL_SCALE = 5e-5


def error_metrics(actual, expected):
    """误差不是一个数：同时给出绝对误差、相对误差、最大误差位置与异常计数。

    - max_abs_err：最坏位置的数值偏差，决定“能不能接受”；
    - max_rel_err：按参考值尺度归一，用来发现小量级位置被淹没；
    - argmax_abs_err：最大误差出现在哪个下标，便于回查具体位置；
    - n_nan / n_inf：普通数值用例里应当为 0，一旦非 0 就说明语义出了问题。

    只报一个“最大误差”会把上面这些信息压扁，因此这里分开报告。
    """
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    abs_err = np.abs(a - e)
    denom = np.maximum(np.abs(e), np.finfo(np.float64).tiny)
    return {
        "max_abs_err": float(abs_err.max()),
        "max_rel_err": float((abs_err / denom).max()),
        "argmax_abs_err": int(np.argmax(abs_err)),
        "n_nan": int(np.isnan(a).sum()),
        "n_inf": int(np.isinf(a).sum()),
    }


def tolerances(ref_sum, ref_scan):
    """教学校验参数：锚定结果尺度，不是所有输入的通用误差界。"""
    reduction_tol = ERROR_TOL_SCALE * max(1.0, abs(float(ref_sum)))
    scan_tol = ERROR_TOL_SCALE * max(1.0, float(np.max(np.abs(ref_scan))))
    return reduction_tol, scan_tol


# ---------- 正确性套件：真实执行，而不是只列清单 ----------

def run_case(host_input, block_size, label):
    n = int(host_input.size)
    validate_config(n, block_size)

    input_gpu = cuda.to_device(host_input)
    reduction_workspace = build_reduction_workspace(n, block_size)
    scan_plan = build_scan_plan(n, block_size)

    try:
        reduced_gpu = launch_global_reduction(
            input_gpu, n, block_size, reduction_workspace
        )
        scan_gpu = launch_global_scan(input_gpu, block_size, scan_plan)

        reduced_host = np.empty(1, dtype=np.float32)
        scan_host = np.empty(n, dtype=np.float32)
        cuda.memcpy_dtoh(reduced_host, reduced_gpu)
        cuda.memcpy_dtoh(scan_host, scan_gpu)

        # float64 参照降低的是“参考端”的舍入误差，它本身不是数学精确真值。
        ref_sum = float(np.sum(host_input, dtype=np.float64))
        ref_scan = np.cumsum(host_input, dtype=np.float64)
        reduction_tol, scan_tol = tolerances(ref_sum, ref_scan)

        red = error_metrics(reduced_host, np.array([ref_sum], dtype=np.float64))
        scn = error_metrics(scan_host, ref_scan)
        passed = (
            red["max_abs_err"] <= reduction_tol
            and scn["max_abs_err"] <= scan_tol
            and red["n_nan"] == 0 and red["n_inf"] == 0
            and scn["n_nan"] == 0 and scn["n_inf"] == 0
        )
    finally:
        # 显式表达资源生命周期：这一轮的设备缓冲区用完即弃，
        # 而不是完全依赖 Python 垃圾回收的隐式时机。
        del input_gpu, reduction_workspace, scan_plan

    print(
        f"[{'PASS' if passed else 'FAIL'}] {label:<14} N={n:<8} "
        f"abs={scn['max_abs_err']:.3e} rel={scn['max_rel_err']:.3e} "
        f"@idx={scn['argmax_abs_err']:<7} "
        f"reduce_abs={red['max_abs_err']:.3e}"
    )
    return passed


def special_equal(actual, expected):
    """NaN 与自身不相等，所以逐位置比较：同位置要么都是 NaN，要么完全相等。"""
    both_nan = np.isnan(actual) & np.isnan(expected)
    equal = ~np.isnan(actual) & ~np.isnan(expected) & (actual == expected)
    return bool((both_nan | equal).all())


def check_single_element_invariants(block_size):
    """N = 1 的边界：workspace 为空、返回值就是输入缓冲区、输入未被改写。"""
    value = np.float32(-3.5)
    host_input = np.array([value], dtype=np.float32)
    input_gpu = cuda.to_device(host_input)
    # n == 1 时不该产生任何归约级
    reduction_workspace = build_reduction_workspace(1, block_size)
    scan_plan = build_scan_plan(1, block_size)

    try:
        ok = len(reduction_workspace) == 0
        reduced_gpu = launch_global_reduction(
            input_gpu, 1, block_size, reduction_workspace
        )
        # 返回值与输入是同一块设备内存，调用者不能假设它总是新分配的
        ok = ok and reduced_gpu is input_gpu

        scan_gpu = launch_global_scan(input_gpu, block_size, scan_plan)
        reduced_host = np.empty(1, dtype=np.float32)
        scan_host = np.empty(1, dtype=np.float32)
        after = np.empty(1, dtype=np.float32)
        cuda.memcpy_dtoh(reduced_host, reduced_gpu)
        cuda.memcpy_dtoh(scan_host, scan_gpu)
        cuda.memcpy_dtoh(after, input_gpu)      # 输入缓冲区当前的内容
    finally:
        del input_gpu, reduction_workspace, scan_plan

    ok = ok and bool(after[0] == value)          # 输入未被改写
    ok = ok and bool(reduced_host[0] == value)   # 归约结果等于输入
    ok = ok and bool(scan_host[0] == value)      # Scan 结果等于输入
    print(
        f"[{'PASS' if ok else 'FAIL'}] N=1              "
        f"workspace 为空 / 返回输入缓冲区 / 输入未被改写"
    )
    return ok



def check_special_value_semantics(block_size):
    """异常值不是精度问题，而是语义问题：用明确的期望序列断言。

    这里验证的是当前编译配置（-O3 + 本文 kernel）下的传播行为。浮点异常的
    传播可能受运算顺序与编译器优化影响，不能推广为“所有优化级别、所有设备、
    所有重排方式都保证完全一致”。

    覆盖范围：Scan 与 Reduction 分开报告。异常语义与数值误差是两套判据，
    这里只用期望序列比对，不使用任何容差。
    """
    nan, inf = float("nan"), float("inf")
    # 每个用例给出完整期望前缀序列；最后一格同时就是 Reduction 的期望结果。
    checks = [
        ("NaN 在首", [nan, 2.0, 3.0, 4.0], [nan, nan, nan, nan]),
        ("NaN 在尾", [1.0, 2.0, 3.0, nan], [1.0, 3.0, 6.0, nan]),
        ("含 +Inf", [1.0, inf, 3.0, 4.0], [1.0, inf, inf, inf]),
        ("含 -Inf", [1.0, -inf, 3.0, 4.0], [1.0, -inf, -inf, -inf]),
        ("+Inf 与 -Inf 相遇", [1.0, inf, -inf, 4.0], [1.0, inf, nan, nan]),
    ]

    results = []
    for label, values, expected_values in checks:
        host_input = np.array(values, dtype=np.float32)
        expected = np.array(expected_values, dtype=np.float32)
        n = int(host_input.size)
        input_gpu = cuda.to_device(host_input)
        reduction_workspace = build_reduction_workspace(n, block_size)
        scan_plan = build_scan_plan(n, block_size)

        try:
            reduced_gpu = launch_global_reduction(
                input_gpu, n, block_size, reduction_workspace
            )
            scan_gpu = launch_global_scan(input_gpu, block_size, scan_plan)

            reduced_host = np.empty(1, dtype=np.float32)
            scan_host = np.empty(n, dtype=np.float32)
            cuda.memcpy_dtoh(reduced_host, reduced_gpu)
            cuda.memcpy_dtoh(scan_host, scan_gpu)
        finally:
            del input_gpu, reduction_workspace, scan_plan

        scan_ok = special_equal(scan_host, expected)
        # 期望值取 expected 的最后一格：它既是 inclusive scan 的终点，也是总和。
        reduce_ok = special_equal(reduced_host, expected[-1:].astype(np.float32))

        results.append(scan_ok and reduce_ok)
        print(
            f"[{'PASS' if scan_ok and reduce_ok else 'FAIL'}] {label:<16} "
            f"Scan={'PASS' if scan_ok else 'FAIL'} "
            f"Reduce={'PASS' if reduce_ok else 'FAIL'}"
        )

    return all(results)


# 快速套件：确认编译、Kernel 启动与基本边界可用
QUICK_SIZES = (1, 7, 256, 257, 1000)
# 完整回归：追加递归路径，以及多组 Block 配置
FULL_EXTRA_SIZES = (65537, 1 << 20)
FULL_BLOCK_SIZES = (32, 64, 128, 256)


def build_distribution_cases(rand):
    """与规模无关的分布用例：它们专门检查累积、抵消与动态范围。"""
    return [
        ("全零", np.zeros(1000, dtype=np.float32)),
        ("全一", np.ones(1000, dtype=np.float32)),
        (
            "正负交替",
            np.array(
                [1.0 if i % 2 == 0 else -1.0 for i in range(1000)],
                dtype=np.float32,
            ),
        ),
        (
            "大小量级混合",
            np.concatenate([rand(500, 1e6), rand(500, 1e-6)]).astype(np.float32),
        ),
    ]


def run_correctness_suite(block_size, mode="full"):
    """mode='quick' 只跑小规模；mode='full' 追加递归、多 block_size 与异常值。"""
    rng = np.random.default_rng(0)

    def rand(n, scale=1.0):
        return (rng.uniform(-1.0, 1.0, size=n) * scale).astype(np.float32)

    sizes = QUICK_SIZES if mode == "quick" else QUICK_SIZES + FULL_EXTRA_SIZES
    cases = [(f"N={n}", rand(n)) for n in sizes]
    cases += build_distribution_cases(rand)

    print(f"=== 正确性套件（{mode}）===")
    results = [run_case(host_input, block_size, label) for label, host_input in cases]

    if mode == "full":
        # Block 大小是本文的核心变量，同一批边界值换几种配置再跑一遍。
        for bs in FULL_BLOCK_SIZES:
            if bs == block_size:
                continue
            for n in (bs - 1, bs, bs + 1, 1000):
                results.append(run_case(rand(n), bs, f"bs={bs}"))
        results.append(check_single_element_invariants(block_size))
        results.append(check_special_value_semantics(block_size))
        results.append(check_pipeline_counts())

    passed = all(results)
    print("=== 套件结果:", "全部通过" if passed else "存在失败用例", "===")
    return passed


def describe_pipeline(n, block_size):
    """数一数教学实现到底启动了多少 Kernel、占了多少中间缓冲区。

    这张“成本清单”用于第 9 节：为什么教学版不够快，不是感觉，而是可数的。
    """
    reduction_levels = 0
    current_n = n
    while current_n > 1:
        reduction_levels += 1
        current_n = ceil_div(current_n, block_size)

    # Scan 的计划树：每一级一个计划，直到 num_blocks == 1。
    scan_levels = 1
    current_n = ceil_div(n, block_size)
    while current_n > 1:
        scan_levels += 1
        current_n = ceil_div(current_n, block_size)

    return {
        "reduction_kernels": reduction_levels,
        "reduction_buffers": reduction_levels,
        "scan_kernels": 2 * scan_levels - 1,   # 每级 1 次局部 Scan + 每级 1 次回填
        "scan_buffers": 2 * scan_levels,       # 每级 output + block_sums
    }


def check_pipeline_counts():
    """把 Kernel 计数规则固定成断言：以后改递归策略，这里会先报警。

    计数规则（与 describe_pipeline 一致）：
    - Reduction：每一级 1 个 Kernel；n == 1 时没有任何归约级，计 0；
    - Scan：每一级计划 1 个局部 Scan Kernel，最底层之外每级再加 1 个回填，
      因此 scan_levels 层对应 2 * scan_levels - 1 个 Kernel。n == 1 仍会
      有一个局部 Scan Kernel，计 1。
    """
    expected = (
        # (n, block_size, reduction_kernels, scan_kernels)
        (1, 256, 0, 1),
        (256, 256, 1, 1),
        (257, 256, 2, 3),
        (1 << 20, 256, 3, 5),
        (1 << 20, 64, 4, 7),
    )
    results = []
    for n, bs, exp_r, exp_s in expected:
        got = describe_pipeline(n, bs)
        good = got["reduction_kernels"] == exp_r and got["scan_kernels"] == exp_s
        results.append(good)
        print(
            f"[{'PASS' if good else 'FAIL'}] counts  n={n:<8} bs={bs:<4} "
            f"reduction={got['reduction_kernels']}(expect {exp_r}) "
            f"scan={got['scan_kernels']}(expect {exp_s})"
        )
    return all(results)



# ---------- Benchmark：GPU Event 时间 ----------

def benchmark(launch_fn, warmup=5, iterations=30):
    """CUDA Event：测量默认 Stream 中 launch_fn 提交的时间区间。

    注意口径：它反映的是 GPU Stream 时间线上的间隔，
    不等同于包含输入准备、显存分配与数据传输的端到端延迟。
    """
    for _ in range(warmup):
        launch_fn()

    cuda.Context.synchronize()
    times_ms = []

    for _ in range(iterations):
        start = cuda.Event()
        end = cuda.Event()

        start.record()
        launch_fn()
        end.record()
        end.synchronize()

        times_ms.append(start.time_till(end))

    times_ms = np.asarray(times_ms, dtype=np.float64)
    return {
        "warmup": warmup,
        "iterations": iterations,
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "min_ms": float(np.min(times_ms)),
        # 把计时口径写进返回值：避免结果被搬到别处后失去上下文
        "reuses_input_buffer": True,     # 重复使用同一块输入，不重新分配
        "includes_kernel_launch": True,  # 区间覆盖多阶段 Kernel 的提交与执行
        "includes_device_alloc": False,  # workspace 已预分配
        "includes_d2h": False,           # 结果不在计时区间内回传
    }


def benchmark_cpu_baseline(host_input, warmup=2, iterations=10):
    """CPU 基线分两组，别把“数值参考”和“性能参考”混为一谈。

    - float64：与正确性检查用的参考序列同精度，属于数值参考；
    - float32：与 GPU 侧同数据类型，才是性能对照的合理起点。

    两者都不包含 H2D / D2H 传输。把 GPU 的 float32 时间直接除以 CPU 的
    float64 时间，得到的不是加速比，只是两个不同问题的耗时之比。
    """
    def timeit(fn):
        for _ in range(warmup):
            fn()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        return (time.perf_counter() - start) / iterations * 1e3

    return {
        "cpu_f32_sum_ms": timeit(lambda: np.sum(host_input, dtype=np.float32)),
        "cpu_f32_scan_ms": timeit(lambda: np.cumsum(host_input, dtype=np.float32)),
        "cpu_f64_sum_ms": timeit(lambda: np.sum(host_input, dtype=np.float64)),
        "cpu_f64_scan_ms": timeit(lambda: np.cumsum(host_input, dtype=np.float64)),
        "note": "NumPy, 不含 H2D/D2H；float32 才与 GPU 侧同精度",
    }


def main():
    parser = argparse.ArgumentParser(description="PyCUDA Reduction / Scan 教学实现")
    parser.add_argument(
        "--test",
        choices=("quick", "full"),
        default="full",
        help="quick: 小规模快速验证；full: 完整回归（默认）",
    )
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    block_size = args.block_size

    if not run_correctness_suite(block_size, mode=args.test):
        raise SystemExit("正确性套件未通过，停止 Benchmark")

    n = 1 << 20
    rng = np.random.default_rng(42)
    host_input = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    input_gpu = cuda.to_device(host_input)

    # 先分配，后执行：计时阶段稳定复用这些缓冲区。
    reduction_workspace = build_reduction_workspace(n, block_size)
    scan_plan = build_scan_plan(n, block_size)

    # Scan 的计时对象是整条设备级流水线：局部 Scan → 递归 Block sums Scan
    # → offset 回填，而不是单个 block_inclusive_scan kernel。
    reduction_stats = benchmark(
        lambda: launch_global_reduction(
            input_gpu, n, block_size, reduction_workspace
        )
    )
    scan_stats = benchmark(
        lambda: launch_global_scan(input_gpu, block_size, scan_plan)
    )

    device = cuda.Context.get_device()
    print("GPU:", device.name())
    print("N:", n, "block_size:", block_size, "test:", args.test)
    print("Pipeline cost:", describe_pipeline(n, block_size))
    print("Reduction timing:", reduction_stats)
    print("Scan timing:", scan_stats)
    print("CPU baseline:", benchmark_cpu_baseline(host_input))

    # 计时之后再抽查一次结果：确保 Benchmark 没有在错误的缓冲区状态上
    # 跑出漂亮的数字。跑得快但算错了，是并行程序里最坏的一种“通过”。
    run_case(host_input, block_size, "计时后抽查")


if __name__ == "__main__":
    main()
