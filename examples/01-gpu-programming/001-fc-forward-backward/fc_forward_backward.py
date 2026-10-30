"""GPU 不认识矩阵：它怎样用 1024 个线程算完一个线性层？（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import pycuda.driver as cuda

# 自动初始化 CUDA Driver、选择默认 GPU 并创建 CUDA 上下文
import pycuda.autoinit

# 导入 NumPy，用于在 CPU 端生成数组和计算参考答案
import numpy as np

# 导入 SourceModule，用于在运行时编译 CUDA C/C++ 源码字符串
from pycuda.compiler import SourceModule


# 使用原始字符串保存 CUDA C/C++ 源码，避免 Python 转义影响源码内容
kernel_code = r"""
// forward：计算单输出线性层 y = Xw
__global__ void forward(
    const float *input,   // 输入矩阵 X，形状为 N×M，只读
    const float *weights, // 权重向量 w，长度为 M，只读
    float *output,        // 输出向量 y，长度为 N，由 Kernel 写入
    int N,                // 样本数量，也就是输入矩阵的行数
    int M)                // 特征数量，也就是输入矩阵的列数
{
    // 计算当前线程在整个一维 Grid 中的全局编号，并把它作为样本行号
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    // 只让拥有合法样本行号的线程继续执行，避免数组越界
    if (row < N) {
        // 使用寄存器中的局部变量保存当前样本点积的累加结果
        float sum = 0.0f;

        // 依次遍历当前样本的全部 M 个特征
        for (int col = 0; col < M; col++) {
            // 读取 X[row,col] 和 w[col]，完成一次乘法并累加到 sum
            sum += input[row * M + col] * weights[col];
        }

        // 当前线程把最终点积写入自己负责的输出位置
        output[row] = sum;
    }
}


// backward_input：计算输入梯度 dL/dX = grad_output × weights^T
__global__ void backward_input(
    const float *grad_output, // 上游梯度 dL/dy，长度为 N，只读
    const float *weights,     // 前向使用的权重向量 w，长度为 M，只读
    float *grad_input,        // 待写入的输入梯度 dL/dX，形状为 N×M
    int N,                    // 样本数量
    int M)                    // 特征数量
{
    // 为 N×M 个输入梯度元素建立一维全局编号
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 计算输入矩阵和输入梯度矩阵的元素总数
    int total = N * M;

    // 过滤向上取整后多启动的线程，避免越界
    if (idx < total) {
        // 用整数除法把一维编号还原为二维行号
        int row = idx / M;

        // 用取模运算把一维编号还原为二维列号
        int col = idx % M;

        // 根据链式法则计算 dL/dX[row,col] = dL/dy[row] × w[col]
        grad_input[idx] = grad_output[row] * weights[col];
    }
}


// backward_weights：计算权重梯度 dL/dw = X^T × grad_output
__global__ void backward_weights(
    const float *input,       // 前向输入矩阵 X，形状为 N×M，只读
    const float *grad_output, // 上游梯度 dL/dy，长度为 N，只读
    float *grad_weights,      // 待写入的权重梯度 dL/dw，长度为 M
    int N,                    // 样本数量
    int M)                    // 特征数量
{
    // 让每个线程负责一个特征列，也就是一个权重梯度元素
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    // 只让列号位于 [0,M) 内的线程继续执行
    if (col < M) {
        // 在寄存器中保存当前权重收到的全部样本贡献
        float sum = 0.0f;

        // 遍历全部 N 条样本，完成当前列上的归约
        for (int row = 0; row < N; row++) {
            // 累加 X[row,col] × dL/dy[row]
            sum += input[row * M + col] * grad_output[row];
        }

        // 当前线程独占 grad_weights[col]，因此不会发生并发写冲突
        grad_weights[col] = sum;
    }
}
"""


# 调用 NVCC 编译 CUDA 源码，并把结果加载为 PyCUDA 模块
module = SourceModule(kernel_code)

# 从编译后的模块中取得名为 forward 的 Kernel 句柄
forward_kernel = module.get_function("forward")

# 从编译后的模块中取得计算输入梯度的 Kernel 句柄
backward_input_kernel = module.get_function("backward_input")

# 从编译后的模块中取得计算权重梯度的 Kernel 句柄
backward_weights_kernel = module.get_function("backward_weights")


# 设置样本数量：一共有 1024 条输入样本
N = 1024

# 设置特征数量：每条样本包含 512 个 float32 特征
M = 512

# 创建固定种子的随机数生成器，保证每次运行都能复现实验数据
rng = np.random.default_rng(42)

# 生成形状为 N×M 的 float32 输入矩阵 X
input_data = rng.random((N, M), dtype=np.float32)

# 生成长度为 M 的 float32 权重向量 w
weights = rng.standard_normal(M).astype(np.float32)

# 为长度为 N 的前向输出 y 分配 CPU 内存，内容稍后由 GPU 覆盖
output_data = np.empty(N, dtype=np.float32)

# 生成长度为 N 的上游梯度 dL/dy，它是反向传播的输入
grad_output = rng.standard_normal(N).astype(np.float32)

# 为真正的输入梯度 dL/dX 分配 N×M 的 CPU 数组
grad_input = np.empty((N, M), dtype=np.float32)

# 为权重梯度 dL/dw 分配长度为 M 的 CPU 数组
grad_weights = np.empty(M, dtype=np.float32)


# 在 GPU 上分配输入显存，并把 input_data 从 CPU 复制到 GPU
input_gpu = cuda.to_device(input_data)

# 在 GPU 上分配权重显存，并把 weights 从 CPU 复制到 GPU
weights_gpu = cuda.to_device(weights)

# 只为前向输出分配 GPU 显存，因为输出会被 Kernel 完整写入
output_gpu = cuda.mem_alloc(output_data.nbytes)

# 在 GPU 上分配上游梯度显存，并把 grad_output 从 CPU 复制到 GPU
grad_output_gpu = cuda.to_device(grad_output)

# 只为输入梯度分配 GPU 显存，因为全部元素会被 backward_input 写入
grad_input_gpu = cuda.mem_alloc(grad_input.nbytes)

# 只为权重梯度分配 GPU 显存，因为全部元素会被 backward_weights 写入
grad_weights_gpu = cuda.mem_alloc(grad_weights.nbytes)


# 设置每个一维线程块包含 256 个线程
block_size = 256

# PyCUDA 使用三元组描述 Block 的 x、y、z 三个维度
block = (block_size, 1, 1)

# 为 N 个前向任务计算一维 Grid 大小，并使用向上取整避免遗漏样本
grid_n = ((N + block_size - 1) // block_size, 1)

# 为 N×M 个输入梯度元素计算一维 Grid 大小
grid_nm = ((N * M + block_size - 1) // block_size, 1)

# 为 M 个权重梯度元素计算一维 Grid 大小
grid_m = ((M + block_size - 1) // block_size, 1)


# 创建记录 GPU 前向起始时刻的 CUDA Event
forward_start = cuda.Event()

# 创建记录 GPU 前向结束时刻的 CUDA Event
forward_end = cuda.Event()

# 把前向开始事件记录到当前 CUDA Stream
forward_start.record()

# 启动前向 Kernel：每个线程负责一条样本的点积
forward_kernel(
    input_gpu,             # 参数1：GPU 上的输入矩阵
    weights_gpu,           # 参数2：GPU 上的权重向量
    output_gpu,            # 参数3：GPU 上的输出数组
    np.int32(N),           # 参数4：以 int32 形式传入样本数量
    np.int32(M),           # 参数5：以 int32 形式传入特征数量
    block=block,           # 指定每个 Block 的线程组织
    grid=grid_n,           # 指定前向计算所需的 Block 数量
)

# 把前向结束事件排在 Kernel 之后
forward_end.record()

# 等待结束事件完成，确保前向 Kernel 已经执行完毕
forward_end.synchronize()

# 计算两个 Event 之间的设备端时间，单位为毫秒
forward_ms = forward_start.time_till(forward_end)


# 创建记录反向传播起始时刻的 CUDA Event
backward_start = cuda.Event()

# 创建记录反向传播结束时刻的 CUDA Event
backward_end = cuda.Event()

# 把反向开始事件记录到当前 CUDA Stream
backward_start.record()

# 启动输入梯度 Kernel：每个线程负责 dL/dX 中的一个元素
backward_input_kernel(
    grad_output_gpu,       # 参数1：GPU 上的上游梯度 dL/dy
    weights_gpu,           # 参数2：GPU 上的权重向量
    grad_input_gpu,        # 参数3：GPU 上待写入的输入梯度
    np.int32(N),           # 参数4：样本数量
    np.int32(M),           # 参数5：特征数量
    block=block,           # 指定每个 Block 的线程组织
    grid=grid_nm,          # 指定覆盖 N×M 个元素的 Grid
)

# 启动权重梯度 Kernel：每个线程负责一个权重梯度元素
backward_weights_kernel(
    input_gpu,             # 参数1：GPU 上的前向输入矩阵
    grad_output_gpu,       # 参数2：GPU 上的上游梯度 dL/dy
    grad_weights_gpu,      # 参数3：GPU 上待写入的权重梯度
    np.int32(N),           # 参数4：样本数量
    np.int32(M),           # 参数5：特征数量
    block=block,           # 指定每个 Block 的线程组织
    grid=grid_m,           # 指定覆盖 M 个权重的 Grid
)

# 把反向结束事件排在两个反向 Kernel 之后
backward_end.record()

# 等待反向结束事件完成，确保两个反向 Kernel 都执行完毕
backward_end.synchronize()

# 计算两个反向 Kernel 的设备端总耗时，单位为毫秒
backward_ms = backward_start.time_till(backward_end)


# 把 GPU 上的前向输出复制回 CPU 的 output_data
cuda.memcpy_dtoh(output_data, output_gpu)

# 把 GPU 上的输入梯度复制回 CPU 的 grad_input
cuda.memcpy_dtoh(grad_input, grad_input_gpu)

# 把 GPU 上的权重梯度复制回 CPU 的 grad_weights
cuda.memcpy_dtoh(grad_weights, grad_weights_gpu)


# 使用 NumPy 矩阵向量乘法计算前向传播参考答案 y = Xw
output_ref = input_data @ weights

# 利用广播计算输入梯度参考答案 dL/dX[n,m] = dL/dy[n] × w[m]
grad_input_ref = grad_output[:, None] * weights[None, :]

# 使用矩阵向量乘法计算权重梯度参考答案 dL/dw = X^T × dL/dy
grad_weights_ref = input_data.T @ grad_output


# 使用相对和绝对容差检查 GPU 前向输出是否接近 NumPy 结果
np.testing.assert_allclose(
    output_data,
    output_ref,
    rtol=1e-4,
    atol=1e-4,
)

# 检查 GPU 输入梯度是否接近 NumPy 广播计算结果
np.testing.assert_allclose(
    grad_input,
    grad_input_ref,
    rtol=1e-4,
    atol=1e-4,
)

# 检查 GPU 权重梯度是否接近 NumPy 矩阵向量乘法结果
np.testing.assert_allclose(
    grad_weights,
    grad_weights_ref,
    rtol=1e-4,
    atol=1e-4,
)


# 打印前向 Kernel 的设备端执行时间
print(f"前向 Kernel 执行时间: {forward_ms:.3f} ms")

# 打印两个反向 Kernel 的设备端总执行时间
print(f"反向 Kernels 执行时间: {backward_ms:.3f} ms")

# 明确告诉读者三组结果已经通过 CPU 参考实现验证
print("前向输出、输入梯度和权重梯度均通过 NumPy 对照验证")

# 打印第一条样本前 5 个输入特征对应的梯度
print("grad_input[0, :5] =", grad_input[0, :5])

# 打印前 5 个权重对应的梯度
print("grad_weights[:5] =", grad_weights[:5])
