"""CSR SpMV：CPU Numba、GPU Numba CUDA，以及 SciPy float64 参考。
依赖：numpy、scipy、numba；GPU 环境另需匹配驱动/Toolkit 的 numba-cuda。
示例：python csr_spmv_corrected.py --rows 100000 --cols 100000 --density 0.0001
"""
import argparse  # 解析命令行参数。
import time  # 用高精度单调时钟进行墙钟计时。
import numpy as np  # 管理主机数组与随机数。
from numba import cuda, njit, config  # CUDA 核函数、CPU JIT 和模拟器配置。
from scipy.sparse import csr_matrix  # 提供独立的稀疏矩阵乘法参考。


@cuda.jit
def csr_spmv_kernel(row_ptr, col_ind, data, x, y):
    """一个线程负责一行；输入值、输出值与累加器均为 float32。"""
    row = cuda.grid(1)  # 全局线程编号 = blockIdx.x * blockDim.x + threadIdx.x。
    if row < y.size:  # 最后一块可能有多余线程，必须进行越界保护。
        acc = np.float32(0.0)  # 显式使用单精度，避免隐式双精度累加。
        start = row_ptr[row]  # 当前行的数据起点（包含）。
        end = row_ptr[row + 1]  # 当前行的数据终点（不包含）。
        for k in range(start, end):  # 仅遍历当前行已存储的元素。
            acc += data[k] * x[col_ind[k]]  # 列索引用于查找 x 的对应分量。
        y[row] = acc  # 空行也会写入 0；每行独立写入，不需要原子操作。


@njit
def csr_spmv_cpu(row_ptr, col_ind, data, x, y):
    """单线程 CPU JIT 基线；复杂度为 O(n_rows + nnz)。"""
    for row in range(y.size):  # 逐行计算，包含空行。
        acc = np.float32(0.0)  # 与 CUDA 保持相同的累加精度。
        for k in range(row_ptr[row], row_ptr[row + 1]):
            acc += data[k] * x[col_ind[k]]  # 只计算 CSR 中存储的条目。
        y[row] = acc  # 覆盖结果，无需提前把 y 清零。


def generate_random_csr_matrix(n_rows, n_cols, density, seed=42):
    """直接生成 CSR，不创建 n_rows × n_cols 稠密矩阵。

    每行条目数服从二项分布，行内列号无放回采样。
    density 是期望非零比例；实际 nnz 会随机波动。
    为简化演示，值使用 float32，索引使用 int32。
    """
    if n_rows < 0 or n_cols < 0:  # 允许零维矩阵，但不允许负维度。
        raise ValueError("矩阵维度不能为负数")
    if not np.isfinite(density) or not 0.0 <= density <= 1.0:
        raise ValueError("density 必须是 [0, 1] 内的有限数")
    limit = np.iinfo(np.int32).max  # int32 索引的可表示上限。
    if max(n_rows, n_cols) > limit:
        raise ValueError("本示例仅支持 int32 范围内的矩阵维度")
    rng = np.random.default_rng(seed)  # 固定种子，便于复现实验。
    counts = rng.binomial(n_cols, density, size=n_rows)  # 每行条目数。
    row_ptr64 = np.empty(n_rows + 1, dtype=np.int64)  # 先用 int64 求前缀和。
    row_ptr64[0] = 0  # CSR 的第一个起点必须为 0。
    np.cumsum(counts, dtype=np.int64, out=row_ptr64[1:])  # 空行对应重复指针。
    nnz = int(row_ptr64[-1])  # 最后一个行指针就是总条目数。
    if nnz > limit:  # 必须在转换 int32 之前检查，防止静默溢出。
        raise ValueError("nnz 超过 int32 上限，请降低规模或改用 int64 索引")
    row_ptr = row_ptr64.astype(np.int32)  # 确认安全后转换为 int32。
    col_ind = np.empty(nnz, dtype=np.int32)  # 只分配实际条目所需的列索引。
    data = np.empty(nnz, dtype=np.float32)  # 只分配实际条目所需的数值空间。
    for row in range(n_rows):  # 逐行生成，临时数组最多为单行规模。
        start, end = int(row_ptr[row]), int(row_ptr[row + 1])
        count = end - start  # 当前行条目数，可能为 0。
        if count == 0:
            continue  # 空行无需填充数据。
        cols = rng.choice(n_cols, size=count, replace=False)  # 防止列号重复。
        cols.sort()  # 行内升序排列，使 CSR 保持规范格式。
        col_ind[start:end] = cols  # 写入当前行列号。
        data[start:end] = rng.uniform(0.1, 1.0, size=count).astype(np.float32)
        # 数值与位置独立生成；采用正值区间避免本例产生显式零元素。
    return data, col_ind, row_ptr  # 不再返回不必要的 dense 矩阵。


def check_result(name, result, reference):
    """对照独立 float64 参考；失败时抛异常，避免带着错误结果报告性能。"""
    error = np.abs(result.astype(np.float64) - reference)  # 用双精度计算误差。
    max_error = float(error.max()) if error.size else 0.0  # 兼容空结果。
    print(f"{name} 最大绝对误差: {max_error:.6e}")
    np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-6)
    # 这是本例数据的工程容差；更长行或强抵消数据需重新评估误差界。


def benchmark_cpu(row_ptr, col_ind, data, x, runs):
    y = np.empty(row_ptr.size - 1, dtype=np.float32)  # 预分配结果。
    csr_spmv_cpu(row_ptr, col_ind, data, x, y)  # 预热，排除首次 JIT 编译。
    start = time.perf_counter()  # CPU 计算使用墙钟计时。
    for _ in range(runs):
        csr_spmv_cpu(row_ptr, col_ind, data, x, y)  # 重用输入与结果数组。
    elapsed = (time.perf_counter() - start) / runs  # 每次调用的平均秒数。
    return y, elapsed


def benchmark_gpu(row_ptr, col_ind, data, x, runs):
    """分别测量驻留 GPU 的重复计算，以及重用显存的往返总耗时。"""
    n_rows = row_ptr.size - 1  # 输出长度由矩阵行数确定。
    if n_rows == 0:
        raise ValueError("零行矩阵不启动 CUDA kernel")
    threads = 256  # 每个线程块 256 个线程。
    blocks = (n_rows + threads - 1) // threads  # 向上取整，覆盖全部行。
    d_row_ptr = cuda.to_device(row_ptr)  # H2D：行指针。
    d_col_ind = cuda.to_device(col_ind)  # H2D：列索引。
    d_data = cuda.to_device(data)  # H2D：非零值。
    d_x = cuda.to_device(x)  # H2D：输入向量。
    d_y = cuda.device_array(n_rows, dtype=np.float32)  # GPU 输出缓冲区。
    y = np.empty(n_rows, dtype=np.float32)  # 预分配主机输出缓冲区。

    def launch():  # 封装启动，重复调用时不再申请设备内存。
        csr_spmv_kernel[blocks, threads](d_row_ptr, d_col_ind, d_data, d_x, d_y)

    launch()  # 预热 CUDA 核函数，排除首次编译及初始化影响。
    cuda.synchronize()  # CPU 继续计时前，等待预热完成。
    if config.ENABLE_CUDASIM:  # 模拟器只能验证逻辑，不能代表 GPU 性能。
        d_y.copy_to_host(y)
        return y, None, None

    begin = cuda.event(timing=True)  # CUDA Event 在设备时间线上记录时间。
    end = cuda.event(timing=True)
    begin.record()  # 默认流开始事件。
    for _ in range(runs):
        launch()  # 数据已驻留 GPU，不包含显存分配和显式数据传输。
    end.record()  # 默认流结束事件，排在最后一次 kernel 后面。
    end.synchronize()  # 必须等事件完成后再读取时间。
    compute_s = cuda.event_elapsed_time(begin, end) / (1000.0 * runs)
    # Event 返回毫秒；此处转换为单次平均秒数，可能包含启动之间的空隙。

    start = time.perf_counter()  # 单独测量含 H2D、计算、D2H 的墙钟耗时。
    for _ in range(runs):
        d_row_ptr.copy_to_device(row_ptr)  # 每次重新传输完整 CSR。
        d_col_ind.copy_to_device(col_ind)
        d_data.copy_to_device(data)
        d_x.copy_to_device(x)
        launch()  # 执行同一个核函数。
        d_y.copy_to_host(y)  # 默认流同步拷回，主机收到完整结果。
    cuda.synchronize()  # 明确确保所有设备工作完成。
    roundtrip_s = (time.perf_counter() - start) / runs
    return y, compute_s, roundtrip_s  # 总耗时不含初始化、编译和缓冲区分配。


def main():
    parser = argparse.ArgumentParser(description="CSR SpMV 正确性与性能演示")
    parser.add_argument("--rows", type=int, default=100000)  # 默认十万行。
    parser.add_argument("--cols", type=int, default=100000)  # 默认十万列。
    parser.add_argument("--density", type=float, default=0.0001)  # 平均每行约 10 项。
    parser.add_argument("--runs", type=int, default=10)  # CPU/GPU 使用相同重复次数。
    parser.add_argument("--seed", type=int, default=42)  # 可复现的随机种子。
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs 必须大于 0")
    data, col_ind, row_ptr = generate_random_csr_matrix(
        args.rows, args.cols, args.density, args.seed
    )
    rng = np.random.default_rng(args.seed + 1)  # 向量使用独立的随机数流。
    x = rng.random(args.cols, dtype=np.float32)  # x 的长度必须等于矩阵列数。
    total = args.rows * args.cols  # 支持非方阵和零维矩阵。
    actual_density = data.size / total if total else 0.0
    csr_bytes = data.nbytes + col_ind.nbytes + row_ptr.nbytes
    print(f"矩阵: {args.rows} × {args.cols}; nnz={data.size:,}")
    print(f"实际密度: {actual_density:.6%}; CSR 大小: {csr_bytes / 2**20:.2f} MiB")

    matrix = csr_matrix((data, col_ind, row_ptr), shape=(args.rows, args.cols))
    reference = matrix.astype(np.float64) @ x.astype(np.float64)  # 独立高精度参考。
    y_cpu, cpu_s = benchmark_cpu(row_ptr, col_ind, data, x, args.runs)
    check_result("CPU", y_cpu, reference)
    print(f"CPU JIT 单线程平均计算耗时: {cpu_s * 1000:.6f} ms")
    print("CPU 前 10 项:", y_cpu[:10])
    if args.rows == 0:  # 避免 blocks=0 的无效启动。
        print("零行矩阵：结果为空，跳过 GPU。")
        return
    if not cuda.is_available():  # 没有 CUDA 时，CPU 校验仍可完成。
        print("CUDA 不可用：跳过 GPU 校验与计时。")
        return
    y_gpu, gpu_s, roundtrip_s = benchmark_gpu(row_ptr, col_ind, data, x, args.runs)
    check_result("GPU", y_gpu, reference)  # GPU 同样直接对照独立参考。
    print("GPU 前 10 项:", y_gpu[:10])
    if gpu_s is None:
        print("CUDA 模拟器模式：仅验证逻辑，不报告 GPU 性能。")
        return
    print(f"GPU 数据驻留时平均计算耗时: {gpu_s * 1000:.6f} ms")
    print(f"GPU H2D + 计算 + D2H 平均耗时: {roundtrip_s * 1000:.6f} ms")
    if gpu_s > 0:
        print(f"计算口径加速比（CPU / GPU）: {cpu_s / gpu_s:.2f}x")
    if roundtrip_s > 0:
        print(f"含传输口径加速比（CPU / GPU）: {cpu_s / roundtrip_s:.2f}x")


if __name__ == "__main__":
    main()  # 仅直接运行脚本时执行，导入时不自动开始测试。
