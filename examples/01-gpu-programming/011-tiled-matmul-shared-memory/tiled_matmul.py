"""从局部乘加到共享内存复用：用 PyCUDA 拆开 GPU 矩阵乘法（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import numpy as np
import pycuda.autoinit          # 负责创建 CUDA Context
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

N = 1024

# 默认使用 16 × 16 = 256 线程的线程块，降低不同设备上的启动门槛。
# 32 × 32 = 1024 线程是"接近资源上限"的扩展实验，不是推荐默认值。
TILE_SIZE = 16
BLOCK_SIZE = (TILE_SIZE, TILE_SIZE, 1)

# ✅ TILE_SIZE 只有这一个配置源：CUDA 源码里的 #define 由下面这行注入。
# 用占位符替换而不是 f-string，是为了避免把 CUDA 源码里的每个 {} 都写成 {{}}。
_KERNEL_TEMPLATE = r"""
#define TILE_SIZE @TILE_SIZE@

// v0：不分块，K 循环里直接读全局内存
__global__ void naive_matmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 本内核没有 __syncthreads()，越界线程可以安全地提前 return
    if (row >= N || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < N; ++k) {
        // 在本文的行主序布局 + Block=(T,T) + threadIdx.x 作为 warp 内
        // 变化最快维度的前提下：对固定的 row、k，A 的多个线程可能读到
        // 同一地址（是否广播由硬件的合并/缓存/广播机制决定）；B 的 col
        // 在 warp 内通常连续，有利于形成合并访问。
        // 结论依赖线程线性化方式与目标架构，不能脱离布局推广。
        acc += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// v1：分块 + 共享内存复用
__global__ void tiled_matmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // 两块静态共享内存：尺寸由编译期宏 TILE_SIZE 决定
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;                 // Block 内的列位置
    int ty = threadIdx.y;                 // Block 内的行位置

    // 本实现的固定契约：blockDim.x == blockDim.y == TILE_SIZE。
    // 因此下面直接用 TILE_SIZE 换算输出坐标是成立的（见主机端 assert）。
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // 线程私有累加器。
    // 编译器通常会尝试把它分配到寄存器，实际分配结果需结合编译输出检查
    // （寄存器不足时可能发生局部内存溢出 spilling）。
    float Cvalue = 0.0f;

    // 沿 K 维度遍历所有 Tile
    // 向上取整，保证 N 不能被 TILE_SIZE 整除时也覆盖完整 K 维度
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {

        int tile_start = m * TILE_SIZE;   // 本轮 K 起点
        int a_col = tile_start + tx;      // A Tile 的列游标
        int b_row = tile_start + ty;      // B Tile 的行游标

        // 每个线程加载一个 A 元素和一个 B 元素；越界位置填 0
        if (row < N && a_col < N) {
            As[ty][tx] = A[row * N + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (b_row < N && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // 屏障位于所有线程都必经的循环路径上。边界外的线程不提前 return，
        // 而是靠“边界位置填 0 + 最终写回保护”处理。
        // 如果只有部分线程到达该屏障，可能造成挂起或非预期行为。
        __syncthreads();

        // 局部乘加：只读共享内存，不再访问全局内存
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        // 第二道屏障：保证本轮对共享内存的**读取阶段结束**后，
        // 才允许进入下一轮并覆盖 As / Bs
        __syncthreads();
    }

    // 只有合法坐标才写回全局内存
    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}
"""

kernel_code = _KERNEL_TEMPLATE.replace("@TILE_SIZE@", str(TILE_SIZE))

module = SourceModule(kernel_code, options=["-O3"])
naive_matmul = module.get_function("naive_matmul")
tiled_matmul = module.get_function("tiled_matmul")


# ============================================================
# 启动前配置校验
# ============================================================
def validate_launch_config(dev, tile_size, dtype=np.float32):
    """做一组"配置级"检查，排除明显的越界配置。

    能力边界（重要）：
      ✅ 能排除：线程总数、Block 各维度、静态共享内存超过设备上限
      ❌ 不能保证：达到理想 Occupancy、寄存器不溢出、内核一定启动成功
    真正的资源占用还取决于寄存器和其他 SM 资源，
    应以编译输出（如 nvcc -Xptxas -v）与 Nsight 的结果为准。
    """
    attr = cuda.device_attribute
    max_threads = dev.get_attribute(attr.MAX_THREADS_PER_BLOCK)
    max_x = dev.get_attribute(attr.MAX_BLOCK_DIM_X)
    max_y = dev.get_attribute(attr.MAX_BLOCK_DIM_Y)
    max_z = dev.get_attribute(attr.MAX_BLOCK_DIM_Z)
    max_shared = dev.get_attribute(attr.MAX_SHARED_MEMORY_PER_BLOCK)

    threads = tile_size * tile_size
    shared_bytes = 2 * tile_size * tile_size * np.dtype(dtype).itemsize

    if threads > max_threads:
        raise ValueError(f"线程总数 {threads} 超过设备上限 {max_threads}")
    if tile_size > max_x or tile_size > max_y:
        raise ValueError(f"Tile {tile_size} 超过 Block X/Y 维度上限 ({max_x}, {max_y})")
    if max_z < 1:
        raise ValueError("设备 Block Z 维度上限异常")
    if shared_bytes > max_shared:
        raise ValueError(f"静态共享内存 {shared_bytes}B 超过设备上限 {max_shared}B")

    print(f"config OK: {threads} threads/block, {shared_bytes}B shared/block")


# 契约：blockDim.x == blockDim.y == TILE_SIZE（内核用 TILE_SIZE 换算 row/col 的前提）
assert BLOCK_SIZE[0] == TILE_SIZE and BLOCK_SIZE[1] == TILE_SIZE, "Block 维度必须等于 TILE_SIZE"
validate_launch_config(cuda.Context.get_device(), TILE_SIZE)

# 输入契约：这是一个“有明确输入契约”的教学内核，不是自动支持任意矩阵的 API
if N <= 0:
    raise ValueError("N 必须为正整数")
if TILE_SIZE <= 0:
    raise ValueError("TILE_SIZE 必须为正整数")

# ---------- 准备数据 ----------
rng = np.random.default_rng(42)
A = rng.random((N, N), dtype=np.float32)      # 行主序，与 row * N + col 一致
B = rng.random((N, N), dtype=np.float32)

if A.dtype != np.float32 or B.dtype != np.float32:
    raise TypeError("当前内核只接受 float32 输入")
if A.shape != (N, N) or B.shape != (N, N):
    raise ValueError("当前示例只支持 N×N 方阵（非方阵会被明确拒绝）")
C_host = np.empty((N, N), dtype=np.float32)   # 主机端缓冲区

A_dev = cuda.mem_alloc(A.nbytes)              # 设备端显存
B_dev = cuda.mem_alloc(B.nbytes)
C_dev = cuda.mem_alloc(C_host.nbytes)
cuda.memcpy_htod(A_dev, A)
cuda.memcpy_htod(B_dev, B)

# ---------- Grid 尺寸 ----------
g = (N + TILE_SIZE - 1) // TILE_SIZE
GRID_SIZE = (g, g, 1)


def launch(kernel):
    # 不显式传 stream：内核与 Event 都使用 PyCUDA 当前的默认执行流，
    # 因此下面测到的是该内核在这条流上的执行区间。
    # 若改用显式 Stream，应让 Event 与被测内核使用同一条 Stream。
    kernel(A_dev, B_dev, C_dev, np.int32(N), block=BLOCK_SIZE, grid=GRID_SIZE)


def bench(kernel, warmup=5, iterations=20):
    """用 CUDA Event 测量内核耗时（毫秒）。

    测量口径：
      - 只含内核执行时间，不含 H2D/D2H 拷贝与内存分配；
      - 内核与 Event 同在默认执行流上；
      - 20 次采样可以算出一个经验 median / p95，但样本量小，
        统计稳定性有限。本文用它演示统计口径，不作为可靠的尾延迟结论。
    跨实现比较时必须保持口径一致。
    """
    for _ in range(warmup):
        launch(kernel)
    cuda.Context.synchronize()

    start, end = cuda.Event(), cuda.Event()
    samples = []
    for _ in range(iterations):
        start.record()
        launch(kernel)
        end.record()
        end.synchronize()
        samples.append(start.time_till(end))

    samples = np.asarray(samples, dtype=np.float64)
    return samples


# ============================================================
# 正确性校验：每个内核单独校验（先于 Benchmark）
# ============================================================
# 把原始 float32 输入转换为 float64 后在 CPU 上算一个较高精度的参考结果。
# 它用于衡量 GPU float32 结果的误差，但**不复刻 GPU 的累加顺序**，
# 也不等于数学意义上的精确结果。
# 注意：这一步会在主机端额外分配 A/B 的 float64 副本与一个
#      float64 结果矩阵（N=1024 时约 8 MB），并占用 CPU 时间。
#      它只担任“数值验收基准”，**不计入下面 CUDA Event 的 kernel-only 计时**，
#      也不应与 GPU 内核时间做端到端对比。
C_ref = A.astype(np.float64) @ B.astype(np.float64)

# 相对误差定义（全文统一）：
#   rel_err = max(|C_test - C_ref|) / max(|C_ref|)
ref_scale = float(np.max(np.abs(C_ref)))


def validate(kernel, name):
    """跑一次内核并将结果与 C_ref 逐元素比较。

    验收条件说明：
      - np.allclose(rtol=1e-3, atol=1e-3) 是**本文的教学验收条件**，
        不是通用的 GEMM 数值验收标准；
      - rtol 与 atol 叠加生效，结果量级在数百时允许的误差远大于 0.001；
      - 同时报告 max_abs_err / rms_err / rel_err 描述误差规模；
      - 正式工程的阈值应结合数据范围、精度模式与业务容忍度确定。
    """
    launch(kernel)
    cuda.Context.synchronize()

    cuda.memcpy_dtoh(C_host, C_dev)
    C64 = C_host.astype(np.float64)          # 升到 float64 再比较
    abs_err = np.abs(C64 - C_ref)
    max_abs_err = float(np.max(abs_err))
    rms_err = float(np.sqrt(np.mean(abs_err ** 2)))
    rel_err = max_abs_err / max(ref_scale, 1e-30)
    ok = np.allclose(C64, C_ref, rtol=1e-3, atol=1e-3)

    print(f"{name:<10} allclose={ok}  max_abs_err={max_abs_err:.3e}  "
          f"rms_err={rms_err:.3e}  rel_err={rel_err:.3e}")
    if not ok:
        raise RuntimeError(f"{name} 正确性校验失败，先排查正确性再谈性能。")
    return C_host.copy()


# 流程：编译 → 配置校验 → naive 校验 → tiled 校验 → warmup → Benchmark
out_naive = validate(naive_matmul, "naive")
out_tiled = validate(tiled_matmul, "tiled")

# 辅助诊断：两个内核算的是同一道题，结果理应接近。
# 但这不是必要的验收条件——两者的编译优化、FMA 生成与浮点执行路径
# 可能不同，都不必与对方逐位一致。各自的正确性以 C_ref 为准。
agree = np.allclose(out_naive.astype(np.float64), out_tiled.astype(np.float64),
                    rtol=1e-4, atol=1e-4)
print(f"naive vs tiled 近似一致（辅助诊断）: {agree}")

# ============================================================
# Benchmark：只测已验证过的内核
# ============================================================
s_v0 = bench(naive_matmul)
s_v1 = bench(tiled_matmul)

# GFLOPS 用标准 GEMM 工作量模型：每个输出元素 N 次乘加，每次乘加计 2 FLOPs。
# 它是**算法工作量模型**，不是硬件实际执行的指令数，
# 只能用于同一输入、同一精度、同一测量口径下的粗略对照。
flops = 2.0 * N ** 3
to_gflops = lambda ms: flops / (ms * 1e-3) / 1e9


def summarize(samples, name):
    med = float(np.median(samples))
    print(f"{name:<10}{med:>12.4f}{float(np.percentile(samples, 95)):>10.4f}"
          f"{float(samples.min()):>10.4f}{to_gflops(med):>10.1f}")
    return med


print(f"\nDevice: {cuda.Context.get_device().name()}")
print(f"N = {N}, TILE_SIZE = {TILE_SIZE}, FLOPs model = 2 * N^3 = {flops:.3e}")
print(f"{'版本':<10}{'median(ms)':>12}{'p95(ms)':>10}{'min(ms)':>10}{'GFLOPS':>10}")
med_v0 = summarize(s_v0, "v0 naive")
med_v1 = summarize(s_v1, "v1 tiled")
print(f"speedup (v0/v1) = {med_v0 / med_v1:.2f}x")
