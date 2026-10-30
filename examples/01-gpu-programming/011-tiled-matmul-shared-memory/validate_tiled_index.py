"""validate_tiled_index.py —— 主机端索引与边界逻辑验证（无需 GPU）

职责边界：
  ✅ 验证索引映射、分块轮数（ceil vs floor）、越界补零
  ❌ 不验证 CUDA 编译 / GPU 运行 / float32 数值行为 / 性能

对应第 4 节 tiled_matmul 内核的取数逻辑：
  第 m 轮取 K 维度 [m*T, m*T+T)，
  As[ty][k] = A[row, m*T+k]，Bs[k][tx] = B[m*T+k, col]，
  越界位置填 0。
"""
import numpy as np

# 显式声明 float64：这里验证的是"索引逻辑"，不是 float32 的舍入行为
DTYPE = np.float64


def tiled_matmul_cpu(A, B, tile_size, rounds_mode="ceil"):
    n = A.shape[0]
    # 关键差异就在这一行：
    #   ceil  → (n + T - 1) // T，覆盖不整除的尾部
    #   floor → n // T，丢掉尾部（用来复现错误写法）
    rounds = (n + tile_size - 1) // tile_size if rounds_mode == "ceil" else n // tile_size

    C = np.zeros((n, n), dtype=DTYPE)
    for m in range(rounds):
        tile_start = m * tile_size
        a_tile = np.zeros((n, tile_size), dtype=DTYPE)          # 显式 dtype
        b_tile = np.zeros((tile_size, n), dtype=DTYPE)          # 显式 dtype
        for k in range(tile_size):
            kk = tile_start + k
            if kk < n:                       # 越界填 0（对应内核的 else 分支）
                a_tile[:, k] = A[:, kk]
                b_tile[k, :] = B[kk, :]
        C += a_tile @ b_tile                 # 本轮局部贡献，累加进结果
    return C


def rel_err(C_test, C_ref):
    """全文统一的相对误差定义：inf 范数下的最大绝对误差 / 参考最大绝对值。"""
    scale = max(np.max(np.abs(C_ref)), 1e-30)
    return float(np.max(np.abs(C_test - C_ref)) / scale)


CASES = [(31, 16), (32, 16), (33, 16), (63, 16), (65, 16),
         (257, 16), (1000, 16), (1024, 16), (1024, 8), (1, 16)]


def random_case(n, t):
    """每个用例使用独立 RNG，避免测试顺序影响输入数据。"""
    case_rng = np.random.default_rng(42 + n + t)
    A = case_rng.random((n, n), dtype=DTYPE)
    B = case_rng.random((n, n), dtype=DTYPE)
    return A, B, A @ B


if __name__ == "__main__":
    print(f"{'用例':<16}{'轮数':>5}{'rel_err(ceil)':>16}{'rel_err(floor)':>17}")
    print("-" * 56)
    for n, t in CASES:
        A, B, ref = random_case(n, t)
        rounds = (n + t - 1) // t
        e_ok = rel_err(tiled_matmul_cpu(A, B, t, "ceil"), ref)
        e_bad = rel_err(tiled_matmul_cpu(A, B, t, "floor"), ref)
        print(f"N={n:<5}T={t:<4}{rounds:>9}{e_ok:>16.2e}{e_bad:>17.2e}")

    # --- 断言 1：正确实现必须与参考一致（随机回归） ---
    for n, t in CASES:
        A, B, ref = random_case(n, t)
        assert rel_err(tiled_matmul_cpu(A, B, t, "ceil"), ref) < 1e-12, (n, t)

    # --- 断言 2：确定性用例，误差可手算，不依赖随机数据 ---
    # A、B 全为 1，N=33、T=16 时，C 的每个元素等于 K=33。
    #   ceil  → 3 轮（覆盖 k=0..31 与 k=32）   → 结果 33，正确
    #   floor → 2 轮（只覆盖 k=0..31，漏掉 k=32） → 结果 32，少 1
    n, t = 33, 16
    ones = np.ones((n, n), dtype=DTYPE)
    ref_ones = ones @ ones                                   # 精确等于 33
    got_ceil = tiled_matmul_cpu(ones, ones, t, "ceil")
    got_floor = tiled_matmul_cpu(ones, ones, t, "floor")
    assert np.all(ref_ones == 33.0), "参考值应为精确的 33"
    assert np.all(got_ceil == 33.0), "ceil 版本应为精确的 33"
    assert np.all(got_floor == 32.0), "floor 版本应为精确的 32（漏掉最后一个 K 元素）"
    print(f"\n确定性用例 N={n}, T={t}（全 1 矩阵）：")
    print(f"  参考值 = {ref_ones[0,0]:.1f}  ceil 版 = {got_ceil[0,0]:.1f}  "
          f"floor 版 = {got_floor[0,0]:.1f}  差 = {ref_ones[0,0] - got_floor[0,0]:.1f}")

    # --- 断言 3：错误写法在非整除用例上应暴露（回归护栏，非普适定理） ---
    # 不假设“误差必然超过某个固定阈值”（某些输入可能让被遗漏的尾部接近零），
    # 只断言“结果不再等于参考值”。真正的证明在上面的确定性全 1 用例。
    for n, t in CASES:
        if n % t == 0:
            continue          # 整除时 ceil 与 floor 等价，本用例无法区分
        A, B, ref = random_case(n, t)
        got = tiled_matmul_cpu(A, B, t, "floor")
        assert not np.allclose(got, ref, rtol=1e-12, atol=1e-12), (n, t)

    print("\n全部断言通过：向上取整正确；floor 在非整除用例上与参考不一致。")
