"""GPU 明明很快，程序为什么还在等？一次讲透 CPU–GPU 协同计算（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：Numba、NumPy。
"""
from time import perf_counter

# 导入 NumPy，用于创建矩阵、生成参考结果和验证数值正确性
import numpy as np

# cuda：编写和启动 GPU 核函数
# njit：将 CPU 数值函数 JIT 编译为本地机器码
# prange：表达可以并行执行的 CPU 循环
# get_num_threads：查询 Numba 当前使用的 CPU 工作线程数量
# float32：在 CUDA 内核中显式创建单精度累加器
from numba import cuda, njit, prange, get_num_threads, float32

# 每个实验重复取样的次数：多次采样取中位数，降低系统噪声
SAMPLES = 15


# 使用 Numba 将 CPU 矩阵加法编译为本地机器码
# parallel=True 允许 prange 循环由多个 CPU 工作线程执行
# cache=True 尝试缓存编译结果，减少后续启动时的编译成本
@njit(parallel=True, cache=True)
def matrix_addition_cpu(A, B, C):
    # 获取矩阵行数
    rows = A.shape[0]

    # 获取矩阵列数
    cols = A.shape[1]

    # 不同工作线程可以并行处理不同的矩阵行
    for row in prange(rows):
        # 遍历当前行中的所有列
        for col in range(cols):
            # 对应元素相加，并写入输出矩阵
            C[row, col] = A[row, col] + B[row, col]


# 使用 Numba CUDA 定义 GPU 矩阵乘法核函数
@cuda.jit
def matrix_multiply_kernel(A, B, C):
    # 获取当前线程在整个二维 Grid 中的全局坐标
    # x 方向映射到矩阵列，y 方向映射到矩阵行
    col, row = cuda.grid(2)

    # 获取输出矩阵的行数和列数
    rows_c = C.shape[0]
    cols_c = C.shape[1]

    # 网格采用向上取整，边缘线程可能越界，因此必须判断
    if row < rows_c and col < cols_c:
        # 创建线程私有的 float32 累加器
        value = float32(0.0)

        # A 的列数等于 B 的行数，即矩阵乘法的公共维度
        for k in range(A.shape[1]):
            # 计算 A 第 row 行与 B 第 col 列的点积
            value += A[row, k] * B[k, col]

        # 每个线程只写一个输出元素，不会发生写冲突
        C[row, col] = value


def main():
    # 设置固定随机种子，使实验可以复现
    np.random.seed(42)

    # 设置方阵维度；可根据显存和实验需求调整
    N = 1024

    # 创建两个 float32 随机输入矩阵
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    # 为 CPU 矩阵加法分配输出数组
    C_add = np.empty_like(A)

    # 如果 CUDA 不可用，给出明确提示并结束程序
    if not cuda.is_available():
        print("CUDA 不可用，请检查 NVIDIA GPU、驱动和 CUDA 环境。")
        return

    # --------------------------------------------------------
    # 1. CPU JIT 预热
    # --------------------------------------------------------

    # 第一次调用触发类型推断和 JIT 编译，不计入正式测试
    matrix_addition_cpu(A, B, C_add)

    # --------------------------------------------------------
    # 2. 配置二维线程块和网格
    # --------------------------------------------------------

    # 每个 Block 包含 16×16=256 个线程
    threads_per_block = (16, 16)

    # x 方向覆盖矩阵列，采用向上取整
    blocks_x = (N + threads_per_block[0] - 1) // threads_per_block[0]

    # y 方向覆盖矩阵行，同样采用向上取整
    blocks_y = (N + threads_per_block[1] - 1) // threads_per_block[1]

    # 组成二维 Grid
    blocks_per_grid = (blocks_x, blocks_y)

    # 创建非默认 CUDA Stream，用于表达独立的异步任务序列
    stream = cuda.stream()

    # --------------------------------------------------------
    # 3. 预分配三个设备数组（不计入传输计时）
    # --------------------------------------------------------

    # 先为 A、B、C 在显存中分配空间
    # 分配是设备内存操作，统计进“输入传输时间”会高估纯复制成本
    A_device = cuda.device_array_like(A)
    B_device = cuda.device_array_like(B)
    C_device = cuda.device_array((N, N), dtype=np.float32)

    # 确保分配完成，让后续 H2D 计时起点干净
    cuda.synchronize()

    # --------------------------------------------------------
    # 4. 测量 H2D 输入传输时间（只包含纯复制）
    # --------------------------------------------------------

    # 记录主机到设备传输的开始时间
    h2d_start = perf_counter()

    # 将 A 和 B 从 CPU 主存复制到已分配的 GPU 显存
    A_device.copy_to_device(A)
    B_device.copy_to_device(B)

    # 确保前面的设备操作完成，使计时边界明确
    cuda.synchronize()

    # 记录主机到设备传输完成时间
    h2d_end = perf_counter()

    # --------------------------------------------------------
    # 5. GPU JIT 预热
    # --------------------------------------------------------

    # 第一次启动 GPU 内核会触发 CUDA JIT 编译
    matrix_multiply_kernel[
        blocks_per_grid,
        threads_per_block,
        stream
    ](A_device, B_device, C_device)

    # 等待预热任务完成，避免把编译开销计入正式结果
    stream.synchronize()

    # --------------------------------------------------------
    # 6. CPU 单独计算基线（多次采样取中位数）
    # --------------------------------------------------------

    # 存放每次 CPU 加法的耗时
    cpu_times = []

    for _ in range(SAMPLES):
        cpu_start = perf_counter()

        # CPU 多线程矩阵加法
        matrix_addition_cpu(A, B, C_add)

        cpu_times.append(perf_counter() - cpu_start)

    # 使用中位数而不是单次结果，降低系统噪声影响
    cpu_time = float(np.median(cpu_times))

    # --------------------------------------------------------
    # 7. GPU 单独计算基线（CUDA Event 测纯内核时间）
    # --------------------------------------------------------

    # 事件对象在循环外创建一次并重复记录，避免每轮新建对象
    # timing=True 显式声明该事件用于计时（也是默认行为）
    event_start = cuda.event(timing=True)
    event_end = cuda.event(timing=True)

    # 存放每次 GPU 纯内核耗时（秒）
    gpu_times = []

    for _ in range(SAMPLES):
        # 在设备端时间线打点：内核前的起点事件
        event_start.record(stream)

        # 被计时的内核
        matrix_multiply_kernel[
            blocks_per_grid,
            threads_per_block,
            stream
        ](A_device, B_device, C_device)

        # 内核后的终点事件
        event_end.record(stream)

        # 等待终点事件完成，再读取设备端耗时
        event_end.synchronize()

        # 官方计时接口 event_elapsed_time 返回毫秒，换算为秒
        gpu_times.append(cuda.event_elapsed_time(event_start, event_end) / 1000.0)

    # 使用中位数作为纯内核典型耗时
    gpu_time = float(np.median(gpu_times))

    # --------------------------------------------------------
    # 8. 串行 vs 重叠：交替采样取中位数
    # --------------------------------------------------------

    # 分别存放串行阶段和重叠阶段的耗时
    serial_times = []
    overlap_times = []

    for _ in range(SAMPLES):
        # --- 串行版本：先 CPU 加法，再 GPU 乘法 ---
        serial_start = perf_counter()

        # CPU 多线程矩阵加法
        matrix_addition_cpu(A, B, C_add)

        # GPU 矩阵乘法
        matrix_multiply_kernel[
            blocks_per_grid,
            threads_per_block,
            stream
        ](A_device, B_device, C_device)

        # 等待 GPU 计算完成
        stream.synchronize()

        serial_times.append(perf_counter() - serial_start)

        # --- 重叠版本：先提交 GPU，再让 CPU 干自己的活 ---
        overlap_start = perf_counter()

        # 提交 GPU 内核；完成 JIT 预热后，
        # 启动通常不会等待内核执行结束，控制权随即返回
        matrix_multiply_kernel[
            blocks_per_grid,
            threads_per_block,
            stream
        ](A_device, B_device, C_device)

        # GPU 工作期间，CPU 同时执行矩阵加法
        matrix_addition_cpu(A, B, C_add)

        # 只有在确实需要 GPU 结果时才同步
        stream.synchronize()

        overlap_times.append(perf_counter() - overlap_start)

    # 取中位数作为典型耗时
    serial_time = float(np.median(serial_times))
    overlap_time = float(np.median(overlap_times))

    # --------------------------------------------------------
    # 9. 将 GPU 结果复制回 CPU
    # --------------------------------------------------------

    # 在主机端分配接收数组
    C_multiply = np.empty((N, N), dtype=np.float32)

    # 记录设备到主机复制开始时间
    d2h_start = perf_counter()

    # 将 GPU 结果复制回 CPU（该 API 同步完成）
    C_device.copy_to_host(C_multiply)

    # 记录复制结束时间
    d2h_end = perf_counter()

    # --------------------------------------------------------
    # 10. 正确性验证
    # --------------------------------------------------------

    # 使用 NumPy 生成矩阵加法参考结果
    C_add_reference = A + B

    # 使用 NumPy/BLAS 生成矩阵乘法参考结果
    C_multiply_reference = A @ B

    # 检查 CPU 并行矩阵加法是否正确（加法误差很小，容差紧）
    add_correct = np.allclose(
        C_add,
        C_add_reference,
        rtol=1e-6,
        atol=1e-6
    )

    # 检查 GPU 矩阵乘法是否在允许的浮点误差范围内
    # 容差不是固定模板，应按数据规模、数值范围和累加顺序确定
    multiply_correct = np.allclose(
        C_multiply,
        C_multiply_reference,
        rtol=1e-3,
        atol=1e-2
    )

    # 计算 GPU 结果的最大绝对误差
    max_error = np.max(
        np.abs(C_multiply - C_multiply_reference)
    )

    # 计算相对 Frobenius 误差：整体误差水平的归一化度量
    relative_error = (
        np.linalg.norm(C_multiply - C_multiply_reference, ord="fro")
        / np.linalg.norm(C_multiply_reference, ord="fro")
    )

    # --------------------------------------------------------
    # 11. 输出测量结果
    # --------------------------------------------------------

    h2d_time = h2d_end - h2d_start
    d2h_time = d2h_end - d2h_start

    print(f"Numba CPU 工作线程数：{get_num_threads()}")
    print(f"H2D 输入传输时间：{h2d_time:.6f} 秒")
    print(f"D2H 结果传输时间：{d2h_time:.6f} 秒")
    print(f"CPU 单独计算时间：{cpu_time:.6f} 秒")
    print(f"GPU 单独计算时间：{gpu_time:.6f} 秒")
    print(f"串行计算阶段时间：{serial_time:.6f} 秒")
    print(f"重叠计算阶段时间：{overlap_time:.6f} 秒")

    # 验证理论关系：串行 ≈ CPU + GPU，重叠 ≈ max(CPU, GPU)
    print(f"[验证] CPU+GPU = {cpu_time + gpu_time:.6f} 秒 vs 串行 {serial_time:.6f} 秒")
    print(f"[验证] max(CPU,GPU) = {max(cpu_time, gpu_time):.6f} 秒 vs 重叠 {overlap_time:.6f} 秒")

    # 计算实际加速比与理论上限
    if overlap_time > 0.0:
        achieved = serial_time / overlap_time
        theoretical = (cpu_time + gpu_time) / max(cpu_time, gpu_time)
        print(f"重叠实际加速比：{achieved:.3f}x")
        print(f"重叠理论上限：{theoretical:.3f}x")

    print(f"CPU 加法结果正确：{add_correct}")
    print(f"GPU 乘法结果正确：{multiply_correct}")
    print(f"GPU 乘法最大绝对误差：{max_error:.6e}")
    print(f"GPU 乘法相对 Frobenius 误差：{relative_error:.6e}")

    # 展示结果矩阵左上角的小块区域，便于人工观察
    print("\n矩阵加法结果前 5 行、前 5 列：")
    print(C_add[:5, :5])

    print("\n矩阵乘法结果前 5 行、前 5 列：")
    print(C_multiply[:5, :5])


# 仅在直接运行当前脚本时调用 main
if __name__ == "__main__":
    main()
