# 一行平方运算，怎样变成 GPU 上的自定义算子？

> **自定义算子 · 第二篇**
>
> `y = x.square()` 背后，GPU 线程怎样找到自己的数据？自己写出的 CUDA 函数，又怎样接住 Python 里的张量？这一篇，我们用平方运算，把中间的连接逐段拆开。
>
> 本文面向有基础 Python 知识的 AI 兴趣者。提供完整的三文件教学工程；代码尚未在目标 GPU 环境编译运行，以下提供实现与验收方法。资料核对日期：2026 年 10 月 19 日。

<!-- more -->

## 01｜先把需求缩小到：每个数字乘自己

先不处理整张图，也不搭神经网络。我们只拿一组数字，要求 GPU 把每个数字平方。计算越简单，越容易看清数据和调用经过了哪些层。

$$
y_i=x_i\times x_i
$$

`i` 是元素的位置。输入 `[1, -2, 3, -4]`，数学上期望得到 `[1, 4, 9, 16]`。

这只是教学练习，不是任何模型的通用预处理步骤。它足够简单，能把注意力留给计算以外的连接。

PyTorch 已经提供 `x.square()`。这一篇自己实现它，目标是**看清一段 CUDA 计算怎样被调用，并验证它是否正确**。它没有需要宣称的性能优势。

上一篇讨论过，算子描述计算含义，内核负责具体执行。本篇先走通一个直接绑定的 CUDA 扩展：

| 本篇完成什么 | 后续继续解决什么 |
| --- | --- |
| 输入张量，返回逐元素平方结果 | 正式算子注册与框架调度 |
| 理解线程索引与边界判断 | 自动微分与训练支持 |
| 编译 C++／CUDA 并从 Python 调用 | 编译器兼容与性能优化 |

**直接绑定能运行，不等于自动具备框架的全部算子能力。**PyTorch 官方明确区分了外部函数绑定与注册算子后接入框架机制的工作。[1]

## 02｜没有显式 for，每个元素是谁算的？

平方的核心计算，其实只有几行：

```cpp
if (i < n) {
    output[i] = input[i] * input[i];
}
```

平方本身已经写完了。真正要回答的是三个问题：`i` 从哪里来，`input` 指向哪里，谁来启动这些线程。这三问会串起全文。

在 CPU 上，你可能用一个循环处理：

```python
for i in range(n):
    output[i] = input[i] * input[i]
```

打个比方：一个人批改一百份卷子，只能一份接一份（CPU 循环）；换成一百个人各批一份，就同时完成了（GPU 多线程）。GPU 的难点不在“同时”，而在给每个执行单元一个明确的编号，让它知道自己负责第几个。

GPU 内核里当然也可以写循环，但这类逐元素计算更常见的做法，是让许多线程各负责一个元素；每个线程只需要知道：“我负责第几个？”

CUDA 把线程组织在线程块（block）中，一次内核启动的线程块组成网格（grid）。先只看一维情况：

| 变量 | 含义 |
| --- | --- |
| `blockIdx.x` | 我属于第几个线程块 |
| `blockDim.x` | 每块有多少个线程 |
| `threadIdx.x` | 我是块内第几个线程 |

这些编号从 0 开始。把块内位置换成整体位置：

$$
i=\text{blockIdx.x}\times\text{blockDim.x}+\text{threadIdx.x}
$$

例如每块 256 个线程，块 1 中的线程 3，就负责元素 `1 × 256 + 3 = 259`。CUDA 官方入门文档采用同类索引方法解释一维逐元素计算。[2]

线程并不是实际同时占满所有硬件执行单元，也不是一个线程永久绑定一个 CUDA Core。硬件会调度执行；这里建立的是任务分配关系。

### 257 个元素，要安排几个块？

每块 256 个线程，需要向上取整：

```cpp
blocks = (n + threads - 1) / threads;
```

当 `n=257`，结果是两个块，共 512 个线程位置。

| 线程块 | 计算出的元素下标 | 需要工作的位置 |
| --- | --- | --- |
| 块 0 | 0～255 | 全部 |
| 块 1 | 256～511 | 只有 256 |

于是，内核必须先检查 `i < n`，再读取或写入数据。否则多出来的线程可能越界。

**多启动少量线程，再由边界判断筛掉多余工作，是处理非整块长度的常见方式。**256 是本例的起点，不是所有内核的最佳参数。

## 03｜完整调用链中，谁负责哪一段？

项目只需要三个文件：

| 文件 | 职责 |
| --- | --- |
| `square.cpp` | 检查输入，将 C++ 接口绑定给 Python |
| `square_kernel.cu` | 分配输出，启动 CUDA 内核 |
| `run_square.py` | 编译加载扩展，调用并验证 |

运行时，从 Python 传入张量，C++ 接口完成检查，CUDA 启动函数再把设备内存指针交给内核。

把“构建”和“每次调用”分开看，初学者不容易混淆：

| 阶段 | 发生什么 |
| --- | --- |
| 构建与加载 | `.cpp`、`.cu` 分别编译，链接成扩展，再加载进 Python |
| 每次调用 | `ext.square(x)` 进入 C++ 接口，分配输出，提交 CUDA 内核，返回张量 |

编译与加载由 PyTorch 的 `torch.utils.cpp_extension.load` 组织，并不是每调用一次 `ext.square` 就重新编译一次。[3]

运行时，真正把内核任务提交给 GPU 的是普通 C++ 启动函数；Python 并不会直接执行 `__global__` 函数。这是整条链路上最容易误解的一环。

这里的“即时编译”是构建本地扩展，与 `torch.compile` 优化计算图是两件事。

## 04｜先写清本版接受什么输入

为了把实现讲透，我们主动限制输入范围：

| 项目 | 本版约定 |
| --- | --- |
| 设备 | CUDA GPU |
| 布局 | 普通稠密、标准连续布局 |
| 数据类型 | float32 |
| 形状 | 标量或任意维张量，允许空张量 |
| 元素数量 | 不超过 2³¹−1，作为本教学版本的范围限制 |
| 输出 | 形状、dtype、设备与输入一致；不修改输入 |
| 梯度 | 不支持，拒绝 `requires_grad=True` 的输入 |

本例按连续存储的元素顺序处理数据，因此先把支持范围限制为连续张量。若要支持其他布局，还需要核对输入访问方式与输出布局，不能默认沿用同一套假设。矩阵转置后，形状看起来正常，实际访问步长却可能改变。

本篇选择拒绝非连续输入，让约束显式可见。调用者需要时可先执行 `x.contiguous()`，但这可能产生额外复制，不能在性能比较中当作免费操作。

## 05｜第一份文件：让 Python 有一个调用入口

把下面内容保存为 `square.cpp`：

```cpp
#include <torch/extension.h>
#include <limits>

// 实现在 .cu 文件中；这是普通 C++ 函数的声明。
at::Tensor square_cuda(const at::Tensor& input);

at::Tensor square(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.layout() == c10::kStrided,
                "input must have strided layout");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(!input.requires_grad(),
                "this tutorial extension does not support autograd");
    TORCH_CHECK(input.numel() <= std::numeric_limits<int>::max(),
                "too many elements for this tutorial implementation");
    return square_cuda(input);
}

// 一个扩展模块只在这里定义一次 Python 绑定入口。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square", &square, "Elementwise square (CUDA, float32)");
}
```

这段代码没有做平方，它负责把不符合约定的输入挡在前面。

本版只实现 float32。先检查 dtype，可以让调用者在进入内核前就得到明确、易懂的错误，也避免把“支持 CUDA”误解为“支持所有 CUDA 数据类型”。

`m.def("square", &square, ...)` 则让 Python 模块获得一个叫 `square` 的函数。这里只是 Python 绑定（通过 `PYBIND11_MODULE` + `m.def`），还没有在框架里正式注册算子。本篇先用 pybind11 打通 Python 到 CUDA 的调用；正式接入框架时，可以根据实现方式选择 Python 侧注册接口或 C++ 注册接口（`TORCH_LIBRARY`／`TORCH_LIBRARY_IMPL`）。下一篇继续沿 C++ 路线，介绍算子定义与设备实现注册。[1][4]

## 06｜第二份文件：线程真正执行平方

把下面内容保存为 `square_kernel.cu`：

```cpp
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void square_kernel(
    const float* input, float* output, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x)
                      * blockDim.x + threadIdx.x;
    if (i < n) {
        const float value = input[i];
        output[i] = value * value;
    }
}

at::Tensor square_cuda(const at::Tensor& input) {
    // 使用输入所在设备，退出函数时恢复原先设备。
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto output = at::empty_like(input);
    const int64_t n = input.numel();
    if (n == 0) {
        return output;  // 空张量不需要启动内核。
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    square_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
```

### 最核心的计算，其实只有几行

`__global__` 表示这是可从主机侧启动、在 GPU 上执行的内核函数。每个线程得到自己的 `i`，边界内的线程读取一个元素并写出它的平方。

这里不需要 `__syncthreads()`：每个输出元素只由一个线程写入，计算也不依赖其他线程的中间结果。

二维输入为什么也能算？因为在约定的标准连续布局下，可以按一段线性数据访问所有元素。输出由 `empty_like` 保留形状信息，内核只需遍历元素位置。

### `data_ptr` 没有把数据从 CPU 再搬一次

`input.data_ptr<float>()` 取得的是张量数据的地址。输入已经位于 GPU 上，内核拿到地址后读取设备内存。把指针作为参数传递，不等于把整个张量重新复制到 GPU。

这也是本例使用 PyTorch 张量分配输出、而非自行 `cudaMalloc` 的原因：让框架继续管理这块内存的生命周期。

### 为什么用 `empty_like`？

`zeros_like` 会先生成全零结果，但平方内核随后会覆盖每个有效位置。本例不需要提前填零。

`empty_like` 中的初值没有保证，所以前提必须成立：**每个输出元素在被读取前，都被正确写入。**若索引遗漏某些位置，就可能读到未初始化内容。

### 另外三行，保证调用方式合理

- `CUDAGuard`：执行时选择输入所在的 CUDA 设备，避免假定所有张量都在 GPU 0。
- `getCurrentCUDAStream()`：使用该设备当前的执行流。执行流可以先理解为 GPU 任务的有序队列；具体调度留到下一篇展开。
- `C10_CUDA_KERNEL_LAUNCH_CHECK()`：检查内核启动后的错误状态。

PyTorch 官方 C++／CUDA 教程也展示了按当前 CUDA stream 启动内核的方式。[4]

启动检查不等于等待整个内核执行完成。CUDA 执行通常是异步的，一些执行阶段错误要在后续同步时才能暴露；因此测试中还需要显式等待。[5]

## 07｜第三份文件：编译、调用、验证

先准备环境：Linux、兼容的 NVIDIA GPU 与驱动、支持 CUDA 的 PyTorch、匹配的 CUDA Toolkit 与 C++ 编译器，以及 Ninja。

仅能运行 PyTorch GPU 运算，不代表已经具备编译扩展的工具链。编译 `.cu` 需要本地 `nvcc`；不要把 PyTorch 使用的 CUDA runtime 与完整开发工具包混为一谈。[3]

可先检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
c++ --version
python -m pip install ninja
```

工具链具体组合请匹配所用 PyTorch 与 CUDA 版本；本文不是针对所有 2.x 版本和编译器的兼容性承诺。

在前两个文件旁保存 `run_square.py`。下面两段共同组成这个文件，按顺序接在一起。

### 加载扩展并检查正常输入

```python
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

if not torch.cuda.is_available():
    raise RuntimeError("需要可用的 CUDA GPU")

root = Path(__file__).resolve().parent
ext = load(
    name="custom_square_demo",
    sources=[str(root / "square.cpp"), str(root / "square_kernel.cu")],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2"],
    verbose=True,
)

torch.manual_seed(0)
cases = [(0,), (1,), (255,), (256,), (257,), (1000,), (2, 3, 5), ()]
for shape in cases:
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    before = x.clone()
    y = ext.square(x)
    torch.cuda.synchronize(x.device)
    torch.testing.assert_close(y, x.square(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(x, before)
    assert (y.shape, y.dtype, y.device) == (x.shape, x.dtype, x.device)
    print("PASS:", shape)

x = torch.tensor([1., -2., 3., -4.], device="cuda")
y = ext.square(x)
torch.testing.assert_close(y, torch.tensor([1., 4., 9., 16.], device="cuda"))
print("平方示例:", y)
```

`load` 返回编译加载后的模块，`ext.square(x)` 调用前面绑定的 C++ 函数。`.cu` 扩展名让构建工具识别 CUDA 源文件，并使用 `nvcc` 编译。[3]

首次运行需要编译，之后构建缓存可避免不必要的重编译；源码或构建条件改变时仍可能重新构建。构建时间与算子运行时间必须分开看。

### 不合法输入，也要验证它会被拒绝

```python
def expect_error(x, message):
    try:
        ext.square(x)
    except RuntimeError as error:
        assert message in str(error), str(error)
        print("REJECT:", message)
    else:
        raise AssertionError("输入本应被拒绝")

expect_error(torch.ones(4), "CUDA tensor")
expect_error(torch.ones(4, device="cuda", dtype=torch.float64), "float32")
expect_error(torch.ones(2, 3, device="cuda").t(), "contiguous")
expect_error(torch.ones(4, device="cuda", requires_grad=True), "autograd")
```

切换到文件所在目录，运行：

```bash
python run_square.py
```

如果是编译资源受限的机器，可用 `MAX_JOBS=1 python run_square.py` 减少并行编译任务数。[3]

运行成功的判定是：正常用例全部通过，示例输出数值对应 `[1, 4, 9, 16]`，四类非法输入均被预期错误拒绝。上述验收方法用于教学；代码尚未在目标 GPU 环境编译运行，实际结果需在 NVIDIA GPU 上确认。

## 08｜为什么测试里特意放了 0、256 和 257？

它们分别瞄准不同风险：

| 输入 | 检查什么 |
| --- | --- |
| 0 个元素 | 空输入是否直接返回，避免启动零个线程块 |
| 1 个元素 | 大部分线程不工作时，边界判断是否正确 |
| 255／256／257 个元素 | 不足一块、刚好一块、跨入下一块 |
| 1000 个元素 | 多块执行且尾块不满 |
| 形状 `(2, 3, 5)` | 线性计算后是否保留多维形状 |
| 形状 `()` | 零维张量，含一个元素，不能误当成空张量 |
| 形状 `(0,)` | 一维空张量，含零个元素，应走“不启动内核”分支 |
| 转置后的张量 | 非连续布局是否按约定拒绝 |

`()` 是零维张量，但包含一个元素；`(0,)` 是一维空张量，包含零个元素。两者只差一个括号，含义完全不同，所以测试里都要覆盖。

这里说的“结果正确”包含两件事：输出数值符合平方公式；输入没有被修改，且输出形状、类型、设备都与约定一致。测试和前面的接口约定因此形成闭环。

如果只用 `[1, 2, 3, 4]` 测试，很可能看不到分块边界问题。反过来，以上通过也只说明覆盖了这些用例，尚不代表完成工业级验证。

一个不用 GPU 就能检查的推理是：当 `n=257` 时，两个块产生下标 0～511，条件 `i<n` 只保留下标 0～256。每个有效下标恰好出现一次。这证明了本例任务划分的基本性质，但无法代替真实编译与设备执行。

## 09｜现在你应该能回答这三个问题

**一行 Python 怎样到达 GPU？**
Python 调用绑定接口，C++ 检查张量，启动函数分配输出并取得设备指针，CUDA 内核（kernel）再按线程索引处理元素。

**为什么计算很短，外围代码却不少？**
一项能被可靠调用的计算，还需要明确输入约束、管理输出、选对设备与执行流，并让错误暴露在合理的位置。这些代码正是“能稳定调用”的条件。

**它能直接放进训练模型吗？**
本版不能。它主动拒绝需要梯度的输入，也没有提供正式算子注册、梯度公式或编译兼容声明。不要为了绕过报错而直接 `detach()`，那会切断梯度路径。

下一篇沿用这段简单内核，解决一个更具体的问题：**从“Python 能调用它”，走到“PyTorch 知道怎样调度它”，还需要补什么？** 下一篇是本系列第三篇，继续沿 C++ 路线介绍算子定义与设备实现注册。

---

## 附录：遇到报错，先判断它发生在哪一层

| 现象 | 优先检查 |
| --- | --- |
| 提示找不到 Ninja 或 nvcc | 构建依赖、PATH、CUDA Toolkit 路径；必要时正确设置 `CUDA_HOME` |
| CUDA 与编译器不兼容 | PyTorch 所需环境、本地 Toolkit 与主机编译器支持范围 |
| `multiple definition` | 是否在多个文件中重复定义 `PYBIND11_MODULE` 或其他符号 |
| 输入被拒绝 | CUDA 设备、float32、连续布局、梯度标志是否符合约定 |
| `no kernel image` 等设备代码错误 | 当前 Toolkit 与构建目标是否支持这张 GPU |
| GPU 内存访问错误 | 检查索引、长度和数据类型；若已触发非法访问，应重启进程再排查 |

定位异步错误时，可以在启动进程前设置 `CUDA_LAUNCH_BLOCKING=1`，帮助错误更接近实际触发位置。它会改变执行行为，用完应移除。[5]

---

## 参考资料

核对日期：2026 年 10 月 19 日。在线文档可能更新，开发时应查阅与本地安装版本匹配的说明。本例使用常规 ATen／LibTorch 扩展接口，不作跨版本二进制兼容承诺。

1. [PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)：直接调用外部计算与正式自定义算子注册的关系。
2. [NVIDIA CUDA Programming Guide — Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)：线程、线程块、索引、边界检查与内核启动。
3. [PyTorch — torch.utils.cpp_extension](https://docs.pytorch.org/docs/stable/cpp_extension.html)：`load`、混合源文件编译、Ninja、CUDA 工具链与构建参数。
4. [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)：CUDA 实现、当前 stream、正式算子注册与其他框架能力。
5. [PyTorch — CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)：设备、执行流、异步执行和调试同步。
