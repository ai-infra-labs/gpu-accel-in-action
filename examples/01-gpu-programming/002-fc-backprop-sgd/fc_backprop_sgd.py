"""反向传播不是魔法：用 PyCUDA 拆开全连接层的 `dX`、`dW` 与 SGD（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401：创建并持有默认 CUDA 上下文

from pycuda.compiler import SourceModule


# ============================================================
# 1. CUDA Kernel
#
# 单样本全连接层：
#
#   x   : [N]
#   W   : [M,N]
#   dY  : [M]
#   dX  : [N]
#
# 数学关系：
#
#   dX = W^T dY
#   dW = dY x^T
#   W  = W - learning_rate * dW
#
# 线程映射：
#
#   thread j -> dX[j] + W[:,j] update
# ============================================================

kernel_code = r"""
__global__ void gradient_update(
    const float* __restrict__ input,       // x[N]；__restrict__ 承诺该指针不与其余三个数组重叠
    float* __restrict__ weights,           // W[M,N]
    const float* __restrict__ grad_output, // dY[M]
    float* __restrict__ grad_input,        // dX[N]
    int N,
    int M,
    float learning_rate)
{
    // 一个线程负责一个输入特征 j
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= N) {
        return;
    }

    // 同一个线程在整列计算中反复使用 x[j]，
    // 显式读入局部变量，便于编译器保存在寄存器中。
    const float xj = input[j];

    // dX[j] 的累加器
    float dx = 0.0f;

    // 遍历 M 个输出神经元
    for (int i = 0; i < M; ++i) {
        const int index = i * N + j;

        // 先保存旧权重：dX 必须使用更新前的 W。
        const float old_weight = weights[index];
        const float dy = grad_output[i];

        // dX[j] += W_old[i,j] * dY[i]
        dx += old_weight * dy;

        // dW[i,j] = dY[i] * x[j]
        // W[i,j]  = W_old[i,j] - lr * dW[i,j]
        weights[index] =
            old_weight - learning_rate * dy * xj;
    }

    // 当前线程独占 grad_input[j]，不存在写冲突。
    grad_input[j] = dx;
}
"""


# 编译并获取 Kernel
module = SourceModule(
    kernel_code,
    options=["-O3"],
)

gradient_update = module.get_function("gradient_update")


# ============================================================
# 2. 实验参数
# ============================================================

N = 1024              # 输入特征数
M = 512               # 输出神经元数
BLOCK_SIZE = 256      # 合理起点，不宣称是所有 GPU 上的最优值
WARMUP = 5
ITERATIONS = 50
LEARNING_RATE = np.float32(0.01)

assert N > 0 and M > 0
assert BLOCK_SIZE > 0

# launch 配置是 device-dependent fact：同一 BLOCK_SIZE 在不同 GPU 上未必合法
dev = cuda.Context.get_device()
max_threads_per_block = dev.get_attribute(
    cuda.device_attribute.MAX_THREADS_PER_BLOCK
)
assert BLOCK_SIZE <= max_threads_per_block, (
    f"BLOCK_SIZE={BLOCK_SIZE} 超过该设备上限 {max_threads_per_block}"
)

grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE


# ============================================================
# 3. 构造输入
# ============================================================

rng = np.random.default_rng(42)

# 单样本输入 x[N]
x = rng.random(N, dtype=np.float32)

# 初始权重 W0[M,N]
W0 = rng.random((M, N), dtype=np.float32)

# 来自后一层或损失函数的输出梯度 dY[M]
dY = rng.random(M, dtype=np.float32)

assert x.shape == (N,)
assert W0.shape == (M, N)
assert dY.shape == (M,)
assert x.flags.c_contiguous
assert W0.flags.c_contiguous
assert dY.flags.c_contiguous
assert x.dtype == np.float32
assert W0.dtype == np.float32
assert dY.dtype == np.float32


# ============================================================
# 4. CPU Reference：只计算一次更新
# ============================================================

# dX = W0^T @ dY
expected_dX = (W0.T @ dY).astype(np.float32)

# dW = dY[:,None] * x[None,:]
expected_dW = (
    dY[:, np.newaxis] * x[np.newaxis, :]
).astype(np.float32)

# W1 = W0 - lr * dW
expected_W = (
    W0 - LEARNING_RATE * expected_dW
).astype(np.float32)


# ============================================================
# 5. Host -> Device
# ============================================================

x_gpu = cuda.to_device(x)
dY_gpu = cuda.to_device(dY)

# W_gpu 是内核原地修改的工作副本。
W_gpu = cuda.to_device(W0)

# W0_gpu 永远保留初始权重，用于每轮恢复状态。
W0_gpu = cuda.to_device(W0)

# mem_alloc 返回 DeviceAllocation，不是 GPUArray。
dX_gpu = cuda.mem_alloc(x.nbytes)


# ============================================================
# 6. Kernel Launch Helper
# ============================================================

def launch_gradient_update():
    gradient_update(
        x_gpu,
        W_gpu,
        dY_gpu,
        dX_gpu,
        np.int32(N),
        np.int32(M),
        LEARNING_RATE,
        block=(BLOCK_SIZE, 1, 1),
        grid=(grid_size, 1, 1),
    )


# ============================================================
# 7. Warmup
#
# Warmup 会修改 W_gpu。
# 正式 benchmark 的每一轮都会先恢复 W0_gpu，
# 因此 warmup 造成的状态不会进入正式测量。
# ============================================================

for _ in range(WARMUP):
    launch_gradient_update()

cuda.Context.synchronize()


# ============================================================
# 8. Benchmark
#
# 每轮：
#
#   reset W0（不计时） -> kernel（计时） -> W1
# ============================================================

times_ms = []
start_event = cuda.Event()
end_event = cuda.Event()

for _ in range(ITERATIONS):
    # 非异步 D2D 复制；恢复操作位于 Event 计时区间之外。
    cuda.memcpy_dtod(W_gpu, W0_gpu, W0.nbytes)

    start_event.record()
    launch_gradient_update()
    end_event.record()

    # time_till() 前必须确保结束事件已经到达。
    end_event.synchronize()

    times_ms.append(
        start_event.time_till(end_event)
    )

times_ms = np.asarray(times_ms, dtype=np.float64)


# ============================================================
# 9. 独立执行一次，用于 Correctness Validation
# ============================================================

cuda.memcpy_dtod(W_gpu, W0_gpu, W0.nbytes)
launch_gradient_update()
cuda.Context.synchronize()


# ============================================================
# 10. Device -> Host
#
# W_gpu 和 dX_gpu 是 Driver API 的 DeviceAllocation，
# 不是 pycuda.gpuarray.GPUArray，因此这里使用 cuda.memcpy_dtoh()。
# 如果改用 GPUArray，则可用其 .get() 方法取回数据。
# ============================================================

dX = np.empty_like(x)
W_result = np.empty_like(W0)

cuda.memcpy_dtoh(dX, dX_gpu)
cuda.memcpy_dtoh(W_result, W_gpu)


# ============================================================
# 11. 数值验证
# ============================================================

RTOL = 2e-5
ATOL = 2e-5

dx_ok = np.allclose(
    dX,
    expected_dX,
    rtol=RTOL,
    atol=ATOL,
)

w_ok = np.allclose(
    W_result,
    expected_W,
    rtol=RTOL,
    atol=ATOL,
)

dx_abs_error = np.max(np.abs(dX - expected_dX))
w_abs_error = np.max(np.abs(W_result - expected_W))

# 分母设置下限，避免参考值接近 0 时相对误差失真或除零。
dx_rel_error = np.max(
    np.abs(dX - expected_dX)
    / np.maximum(np.abs(expected_dX), np.float32(1e-6))
)

w_rel_error = np.max(
    np.abs(W_result - expected_W)
    / np.maximum(np.abs(expected_W), np.float32(1e-6))
)


# ============================================================
# 12. Benchmark 统计
# ============================================================

min_ms = float(np.min(times_ms))
median_ms = float(np.median(times_ms))
p95_ms = float(np.percentile(times_ms, 95))
max_ms = float(np.max(times_ms))


# ============================================================
# 13. 输出环境与结果
# ============================================================

device = cuda.Context.get_device()

print("=" * 72)
print("PyCUDA 单样本全连接层反向传播 + 原地 SGD")
print("=" * 72)
print(f"GPU               : {device.name()}")
print(f"Compute Capability: {device.compute_capability()}")
print(f"N                 : {N}")
print(f"M                 : {M}")
print(f"Block Size        : {BLOCK_SIZE}")
print(f"Grid Size         : {grid_size}")
print(f"Learning Rate     : {float(LEARNING_RATE)}")

print()
print("Tensor Shapes")
print("-" * 72)
print(f"x                 : {x.shape}")
print(f"W                 : {W0.shape}")
print(f"dY                : {dY.shape}")
print(f"dX                : {dX.shape}")

print()
print("Kernel Mapping")
print("-" * 72)
print("thread j -> dX[j] + W[:,j] update")

print()
print("Benchmark: kernel only, reset excluded")
print("-" * 72)
print(f"Warmup            : {WARMUP}")
print(f"Iterations        : {ITERATIONS}")
print(f"Min               : {min_ms:.6f} ms")
print(f"Median            : {median_ms:.6f} ms")
print(f"P95               : {p95_ms:.6f} ms")
print(f"Max               : {max_ms:.6f} ms")

print()
print("Correctness")
print("-" * 72)
print(f"dX                : {'PASS' if dx_ok else 'FAIL'}")
print(f"dX max abs error  : {dx_abs_error:.8e}")
print(f"dX max rel error  : {dx_rel_error:.8e}")
print(f"W update          : {'PASS' if w_ok else 'FAIL'}")
print(f"W max abs error   : {w_abs_error:.8e}")
print(f"W max rel error   : {w_rel_error:.8e}")

print()
print("Samples")
print("-" * 72)
print("dX[:5] =", dX[:5])
print("W_result[0, :5] =", W_result[0, :5])

if not (dx_ok and w_ok):
    raise RuntimeError(
        "CUDA result does not match the NumPy reference."
    )

print()
print("Validation: PASS")
