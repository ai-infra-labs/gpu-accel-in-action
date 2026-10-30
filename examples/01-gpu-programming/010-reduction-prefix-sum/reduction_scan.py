"""GPU 不只是“同时算”：为什么线程越多，归约和前缀和反而越难？（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import numpy as np

import pycuda.autoinit  # 创建 CUDA Context
import pycuda.driver as cuda
from pycuda.compiler import SourceModule



CUDA_SOURCE = r"""
extern "C" __global__ void block_reduce_sum(
    const float* input,
    float* partial_sums,
    int n)
{
    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;
    const unsigned int index = blockIdx.x * blockDim.x + tid;

    // 尾块不足时补 0；所有线程仍须参加后续屏障。
    sdata[tid] = (index < n) ? input[index] : 0.0f;
    __syncthreads();

    // 折半树要求 blockDim.x 为 2 的幂。
    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0];
}


extern "C" __global__ void block_inclusive_scan(
    const float* input,
    float* output,
    float* block_sums,
    int n)
{
    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;
    const unsigned int index = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (index < n) ? input[index] : 0.0f;
    __syncthreads();

    // Hillis–Steele：每轮把可见范围扩大一倍，同样要求 2 的幂。
    for (unsigned int stride = 1;
         stride < blockDim.x;
         stride <<= 1)
    {
        float left = 0.0f;

        // 先把本轮需要的旧值读进寄存器，避免同轮读写混在一起。
        if (tid >= stride)
            left = sdata[tid - stride];

        __syncthreads();

        if (tid >= stride)
            sdata[tid] += left;

        __syncthreads();
    }

    if (index < n)
        output[index] = sdata[tid];

    // 依赖“补加法单位元 0”这一前提：完整 scan 会把它累加到最后一个槽位。
    if (tid == blockDim.x - 1)
        block_sums[blockIdx.x] = sdata[tid];
}


extern "C" __global__ void add_block_offsets(
    float* output,
    const float* block_prefix,
    int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
        return;

    // block_prefix 是 inclusive 的：前序 Block 的累计和是 block_prefix[b-1]。
    if (blockIdx.x > 0)
        output[index] += block_prefix[blockIdx.x - 1];
}
"""


module = SourceModule(CUDA_SOURCE, options=["-O3"], no_extern_c=True)

block_reduce_sum = module.get_function("block_reduce_sum")
block_inclusive_scan = module.get_function("block_inclusive_scan")
add_block_offsets = module.get_function("add_block_offsets")

FLOAT_BYTES = np.dtype(np.float32).itemsize


def ceil_div(a, b):
    return (a + b - 1) // b


def validate_config(n, block_size):
    """检查本文实现的“配置级”前提。

    以下接口前提由调用者保证，本文不代为检查——它们属于 API 契约，
    而不是 Kernel 配置：
    - input 是一维、C-contiguous 的 float32 设备缓冲区；
    - 该缓冲区的实际容量不小于 n 个 float32；
    - block_size 同时满足设备线程上限与本文两个 Kernel 的资源需求。

    递归会产生 Σ ceil(n / B^k) 量级的中间缓冲区；n 很大时要一并考虑显存占用。
    """
    if n < 1:
        raise ValueError("n 必须大于等于 1")

    # 本文两个 Kernel 都依赖 2 的幂：归约的折半树、Scan 的倍增步长。
    if block_size < 1 or (block_size & (block_size - 1)) != 0:
        raise ValueError("本教学版的 reduction 与 scan 都要求 block_size 是 2 的幂")

    device = cuda.Context.get_device()
    max_threads = device.get_attribute(
        cuda.device_attribute.MAX_THREADS_PER_BLOCK
    )
    max_shared = device.get_attribute(
        cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
    )

    if block_size > max_threads:
        raise ValueError(f"block_size 超过设备上限 {max_threads}")

    # 检查设备报告的每个 Block Shared Memory 上限。
    # 这不等于当前 Kernel 的全部资源约束：它没有算上静态 Shared Memory、
    # 寄存器与 occupancy 的影响。若要用远超默认值的动态 Shared Memory，
    # 还需按运行时要求做额外的 opt-in 配置并单独验证。
    if block_size * FLOAT_BYTES > max_shared:
        raise ValueError("动态 Shared Memory 超过设备报告的每 Block 上限")


# ---------- Reduction：预分配与执行分离 ----------

def build_reduction_workspace(n, block_size):
    """每一级保存一个 partial-sums 缓冲区。"""
    buffers = []
    current_n = n

    while current_n > 1:
        num_blocks = ceil_div(current_n, block_size)
        buffers.append(cuda.mem_alloc(num_blocks * FLOAT_BYTES))
        current_n = num_blocks

    return buffers


def launch_global_reduction(input_gpu, n, block_size, workspace):
    """提交归约 Kernel，返回含最终结果的设备缓冲区。

    注意返回值的语义：当 n > 1 时，返回 workspace 最后一级（只含 1 个元素）；
    当 n == 1 时没有任何归约级，直接返回 input_gpu。
    因此返回值**可能与输入缓冲区相同**，调用者不应假设它总是新分配的结果。
    workspace 必须由调用者持续持有，直到不再读取返回值为止。
    """
    current_input = input_gpu
    current_n = n

    for output_gpu in workspace:
        num_blocks = ceil_div(current_n, block_size)

        block_reduce_sum(
            current_input,
            output_gpu,
            np.int32(current_n),
            block=(block_size, 1, 1),
            grid=(num_blocks, 1, 1),
            shared=block_size * FLOAT_BYTES,
        )

        current_input = output_gpu
        current_n = num_blocks

    # n == 1 时 workspace 为空，直接返回输入指针。
    return current_input


# ---------- Scan：递归 workspace 树 ----------

def build_scan_plan(n, block_size):
    """构建并持有每一级 Scan 的输出、Block sums 和子计划。"""
    num_blocks = ceil_div(n, block_size)

    plan = {
        "n": n,
        "num_blocks": num_blocks,
        "output": cuda.mem_alloc(n * FLOAT_BYTES),
        "block_sums": cuda.mem_alloc(num_blocks * FLOAT_BYTES),
        "child": None,
    }

    if num_blocks > 1:
        plan["child"] = build_scan_plan(num_blocks, block_size)

    return plan


def launch_global_scan(input_gpu, block_size, plan):
    """递归提交：局部 Scan → Block sums Scan → 偏移回填。"""
    n = plan["n"]
    num_blocks = plan["num_blocks"]

    block_inclusive_scan(
        input_gpu,
        plan["output"],
        plan["block_sums"],
        np.int32(n),
        block=(block_size, 1, 1),
        grid=(num_blocks, 1, 1),
        shared=block_size * FLOAT_BYTES,
    )

    if plan["child"] is not None:
        block_prefix_gpu = launch_global_scan(
            plan["block_sums"],
            block_size,
            plan["child"],
        )

        add_block_offsets(
            plan["output"],
            block_prefix_gpu,
            np.int32(n),
            block=(block_size, 1, 1),
            grid=(num_blocks, 1, 1),
        )

    return plan["output"]


def main():
    """最小可运行入口：算一次 Reduction 与 Scan 并打印结果。

    完整的正确性套件与 Benchmark 见 verify_reduction_scan.py（附录 A）。
    """
    n = 1 << 20
    block_size = 256
    validate_config(n, block_size)

    rng = np.random.default_rng(42)
    host_input = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    input_gpu = cuda.to_device(host_input)

    reduction_workspace = build_reduction_workspace(n, block_size)
    scan_plan = build_scan_plan(n, block_size)

    reduced_gpu = launch_global_reduction(
        input_gpu, n, block_size, reduction_workspace
    )
    scan_gpu = launch_global_scan(input_gpu, block_size, scan_plan)

    reduced_host = np.empty(1, dtype=np.float32)
    scan_host = np.empty(n, dtype=np.float32)
    cuda.memcpy_dtoh(reduced_host, reduced_gpu)
    cuda.memcpy_dtoh(scan_host, scan_gpu)

    print("GPU:", cuda.Context.get_device().name())
    print("N:", n, "block_size:", block_size)
    print("GPU reduction:", float(reduced_host[0]))
    print("FP64 reference:", float(np.sum(host_input, dtype=np.float64)))
    print("Scan head:", scan_host[:8])


if __name__ == "__main__":
    main()
