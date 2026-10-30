# 有了 Triton，自定义算子还要写 C++ 吗？

> **自定义算子 · 第八篇**
>
> 手写 GPU 计算，可以少写一些 C++ 和构建代码，但不会因此免去并行设计、接口约定和正确性验证。用 Triton 重写一段熟悉的运算，看看开发方式究竟改变了什么。
>
> 本文面向具备基础 Python 知识的 AI 兴趣者，提供完整三文件工程。创作环境未安装 PyTorch 或 Triton，未执行 GPU 编译、数值测试或性能测量；所有测试和跑分代码均为待在目标环境执行的方法。资料核对日期：2026 年 10 月 19 日。

<!-- more -->

## 01｜计算只有几行，为什么工程里有那么多文件？

先直接回答标题里的问题：**对本文这种 PyTorch 中的逐元素计算，可以用 Triton 完成内核开发与框架接入，不必自己编写 C++ 扩展；已有 C++ 工程、依赖特殊底层功能或受部署约束时，则需要另行评估。**

需要区分两件事：“不必自己编写 C++”不等于“底层没有 C++”。设备代码依然由编译器生成，驱动、平台与版本依赖也依然存在——本篇后面会反复回到这一点。

前几篇，我们为一个很小的 GPU 计算准备了不少外围工作：C++ 接口、设备实现注册、CUDA 文件、编译加载，以及梯度和图分析支持。这些工作各有作用，但对想快速尝试一种计算方式的人来说，成本确实存在。

现在回到摄像头的教学场景。假设一批像素或特征已经位于 GPU，需要执行第五篇的变换：

$$
y_i=\max(2x_i-1,0)
$$

PyTorch 原生表达式是：

```python
y = torch.relu(x * 2.0 - 1.0)
```

第五篇把乘法、加法和 ReLU 合进一个 CUDA 内核，研究中间张量与启动开销。本篇保持数学问题不变，改用 **Triton** 表达 GPU 计算。

这里的 Triton 是 GPU 编程语言与编译工具链，不是同名的推理服务产品 Triton Inference Server。它允许用嵌在 Python 中的专用语言描述 GPU 计算，再由编译器生成设备代码。[1][2]

两条路线各自的活并没有凭空消失，只是分给了不同的角色：

| 工作 | CUDA C++ 路线 | 本文 Triton 路线 |
| --- | --- | --- |
| 表达元素计算 | 显式线程索引 | 逻辑数据块与 mask |
| 构建设备代码 | 编写扩展构建流程 | 由 Triton JIT 处理 |
| PyTorch 接口 | C++ 注册及加载 | `triton_op` 包装 |
| 梯度与验证 | 需要提供 | 仍然需要提供 |

这张表也解释了本篇相对第五篇的“新意”：**文件更少，只是工作组织发生变化的一个结果**，而不是这些问题消失了。

## 02｜最重要的变化：从一个线程，转向一组数据

CUDA 入门版本常从一个线程处理一个元素开始：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    // 读取并处理第 i 个元素。
}
```

Triton 的基础表达方式不同：一个 program instance 处理一块逻辑数据。这里的 program 不是一个 Python 进程，也不是一个 GPU 线程。

例如每个 program 处理 1024 个位置：

```python
offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

`tl.arange(0, 1024)` 表示这块数据中的 1024 个逻辑位置；加上块起点，就得到本 program 处理的位置集合。

| program 编号 | 当 BLOCK_SIZE=1024 时的逻辑位置 |
| --- | --- |
| 0 | 0～1023 |
| 1 | 1024～2047 |
| 2 | 2048～3071 |

实际的数据布局和线程分工由编译器结合配置安排。**`BLOCK_SIZE=1024` 表示处理 1024 个逻辑元素，不等于启动 1024 个线程。**官方向量加法教程用同类分块方式介绍这一模型。[1]

本例还设置了 `num_warps=4`。在本例的 NVIDIA 配置中，`num_warps=4` 表示每个 program 使用 4 个线程束（warp），即 128 个线程协作（`4 × 32`）；而 `BLOCK_SIZE=1024` 描述这组线程负责的逻辑元素数量，具体元素分配由编译器安排。[8] 不要把 CUDA 中的“每块线程数”直接抄成 Triton 的“每 program 元素数”——两者是不同维度的配置。

## 03｜边界检查还在，只是变成了 mask

如果输入有 1025 个元素，需要两个 program。第二个 program 里，只有第一个位置有效。

Triton 使用 mask 筛出有效位置：

```python
mask = offsets < n
values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
tl.store(y_ptr + offsets, result, mask=mask)
```

`load` 不读取越界位置，`store` 不写入越界位置。`other=0.0` 为未加载位置提供替代值，但不能因此省掉写入时的 mask。

再看一个便于手算的小例子。假设 `BLOCK_SIZE=8`、输入长度为 10：第一个 program 处理位置 0～7，第二个处理 8～15；第二个 program 的 mask 只允许位置 8 和 9 参与读和写，位置 10～15 既不读也不写。实际工程为了摊薄启动开销仍采用 1024，这个小尺寸只是帮助理解逻辑块与 mask 的配合，不必改回代码。

逻辑上，mask 与 CUDA 的 `if (i < n)` 处理的是同一类问题。换了表达方式，内存边界并没有消失。[1][3]

本例适合 Triton，是因为各元素相互独立，数据访问连续，计算可以自然地描述成一块输入变成一块输出。更复杂的归约、矩阵分块与布局优化，还需要继续学习对应机制。

## 04｜`triton.jit` 不等于 `torch.compile`

这两个编译入口容易混淆：

| 工具 | 主要处理什么 |
| --- | --- |
| `@triton.jit` | 把用 Triton 语言描述的内核编译成可执行设备代码 |
| `torch.compile` | 捕获并优化 PyTorch 程序中的计算，使用相应后端执行 |

它们可以配合，也可以分别使用。Triton 内核的 Python 语法外观，不表示其中可以任意调用 Python、NumPy 或文件操作；设备计算仍受这套语言与编译器的规则约束。[2]

同样，首调用可能发生 JIT 编译或缓存相关工作，不能把它直接算进稳态内核性能，再与已经预热的原生实现比较。

## 05｜接入 PyTorch，为什么还要选择接口？

对于简单用途，可以直接从 Python 启动 Triton 内核。官方资料建议先采用足够简单的集成方式；需要更结构化的框架协作时，再考虑注册包装。[4]

本系列已经讨论过梯度、元数据和编译协作，因此本篇选择：

- `torch.library.triton_op`：为 Triton 支撑的计算提供正式算子接口。
- `torch.library.wrap_triton`：包装其中的 Triton 内核调用，便于相应图追踪机制识别。
- `register_autograd`：明确提供一阶反向公式。

官方教程注明，`torch.library.triton_op` 在 PyTorch 2.6 中加入；`wrap_triton`、`register_autograd` 及它们的组合行为，仍需按你实际选择的版本逐一验证。[4] 环境准备时按所选 PyTorch 的依赖约束安装匹配版本，不要为了拿到某个新接口就直接覆盖成任意最新版 Triton。

这与第六篇普通外部算子的接入有一个区别：`triton_op` 的实现对 `torch.compile` 可见，而普通 `custom_op` 通常作为不透明调用处理。可见意味着编译器有进一步处理的机会，不保证与周围代码一定融合或一定更快。[4][5]

本例输出分配等逻辑通过可追踪的 PyTorch 操作表达，内核调用用 `wrap_triton` 包装，因此不再手写第六篇那样的独立 fake 实现。要强调的是，这份便利来自**本文所选集成方式与相应版本**，并不是所有 Triton 包装方式都自动免去元数据适配；外部内核该有的输入、输出与设备约定依然要交代清楚。

## 06｜完整实现：这次只需要一个算子文件

建立新目录，保存三个文件：

| 文件 | 内容 |
| --- | --- |
| `affine_triton.py` | Triton 内核、算子包装与反向注册 |
| `test_affine_triton.py` | 数值、梯度、注册、编译和 stream 测试 |
| `bench_affine_triton.py` | 原生、Triton 与编译实现的统一口径比较 |

本例面向 NVIDIA CUDA 环境：标准连续稠密输入，float32 或 float64，最多 2³¹−1 个元素，允许空张量、标量与多维张量；输出新分配。测试使用有限、适中数值，不覆盖 NaN、无穷、溢出、混合精度、高阶梯度或所有动态形状。

下面三段按顺序组成 `affine_triton.py`；测试与跑分的完整代码见附录 A。

### 写出一块数据上的计算

```python
import torch
import triton
import triton.language as tl

for api in ("triton_op", "wrap_triton"):
    if not hasattr(torch.library, api):
        raise RuntimeError(f"当前 PyTorch 缺少 torch.library.{api}")

from torch.library import triton_op, wrap_triton

@triton.jit
def affine_relu_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    value = x * 2.0 - 1.0
    result = tl.where(value > 0.0, value, 0.0)
    tl.store(y_ptr + offsets, result, mask=mask)
```

`BLOCK_SIZE: tl.constexpr` 表示这个参数用于编译期确定内核结构，比如逻辑块的大小；它不意味着可以把所有运行时值都随意当常量。

读取、乘法、减法和判断在一个内核中表达，没有先生成两个完整的 PyTorch 中间张量。不过，实际编译指令和浮点舍入仍需按结果验证，不宣称逐位等同于每种其他实现。

### 为算子写清输入约束和输出分配

```python
@triton_op("triton_lab8::affine_relu", mutates_args={})
def affine_relu(x: torch.Tensor) -> torch.Tensor:
    torch._check(x.device.type == "cuda", lambda: "x must be CUDA")
    torch._check(x.layout == torch.strided, lambda: "use strided layout")
    torch._check(x.is_contiguous(), lambda: "x must be contiguous")
    torch._check(x.dtype in (torch.float32, torch.float64),
                 lambda: "use float32 or float64")
    torch._check(x.numel() <= 2**31 - 1, lambda: "too many elements")
    y = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return y
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch.cuda.device(x.device):
        wrap_triton(affine_relu_kernel)[grid](
            x, y, n, BLOCK_SIZE=1024, num_warps=4
        )
    return y
```

`mutates_args={}` 声明不修改输入。新分配的 y 可以由内核写入，不代表 x 也可以被原地改写。

`triton.cdiv` 向上取整，决定需要多少个 program；空输入则不启动内核。设备上下文选择输入所在 GPU，执行仍需遵守 PyTorch 与 Triton 的 stream 交互约定，后面提供非默认 stream 的检查。

这里使用固定配置作为教学起点，没有自动搜索最优 `BLOCK_SIZE` 或 `num_warps`。参数更大不代表一定更快，资源占用和实际工作负载都需要测量。

### 注册一阶反向公式

```python
def setup_context(ctx, inputs, output):
    x, = inputs
    ctx.save_for_backward(x)

def backward(ctx, grad_output):
    x, = ctx.saved_tensors
    active = x * 2.0 - 1.0 > 0
    slope = active.to(dtype=x.dtype) * 2.0
    return grad_output * slope

affine_relu.register_autograd(backward, setup_context=setup_context)
```

当 `2x−1>0`，局部斜率为 2；当 `2x−1<0`，斜率为 0。在交界 `x=0.5`，本例约定反向斜率为 0，与参照中的 ReLU 反向行为对齐。

外部 GPU 内核并不会因为写成 Triton 就自动获得你想要的数学梯度。官方教程也明确展示了为 `triton_op` 注册反向公式的方式。[4] 这就是对照表里“梯度与验证——仍然需要提供”的含义：换了表达方式，这一环不能省。

## 07｜测试：少了 C++，这些检查仍然要做

C++ 绑定省掉了，但检查没有减少。完整测试脚本见附录 A，这里说明它覆盖什么、想回答什么问题：

- **数值与输入梯度**：多种 dtype 与形状，与原生表达式按容差逐项比较；边界形状（0、1、1023、1024、1025）分别检查不足一块、刚好一块、进入下一块；空张量与标量也保留。
- **数值梯度与注册交互**：用 `gradcheck` 在非光滑点之外核对数值梯度，用 `opcheck` 检查注册接口。两者回答不同问题，不能互相替代。
- **编译路径**：确认 `triton_op` 在 `torch.compile` 下仍能得到正确的前向与一阶反向。
- **非默认 stream**：确认在另一条 stream 上排队的连续任务能正确完成。

编译路径这一项值得单独说。只比较聚合后的平方和损失，会把整个输出张量压成一个标量，掩盖逐元素差异；更直接的做法是让编译函数原样返回 `affine_relu(x)`，直接比较输出张量，并用非全 1 的上游梯度比较反向结果。输入也要确定地覆盖激活与未激活两侧，否则整段落在未激活区、输出与梯度全为零，测试信息就很少：

```python
compiled_affine = torch.compile(affine_relu, backend="inductor",
                                fullgraph=True, dynamic=False)

x = torch.tensor([-1., 0.25, 0.5, 0.75, 2.], device="cuda", requires_grad=True)
xr = x.detach().clone().requires_grad_(True)
g = torch.tensor([1., 2., 3., 4., 5.], device="cuda")
y, yr = compiled_affine(x), reference(xr)
torch.testing.assert_close(y, yr, rtol=1e-5, atol=1e-6)   # 直接比较输出张量
gy, = torch.autograd.grad(y, (x,), grad_outputs=g)
gr, = torch.autograd.grad(yr, (xr,), grad_outputs=g)
torch.testing.assert_close(gy, gr, rtol=1e-5, atol=1e-6)  # 非全 1 上游梯度
print("PASS compiled full output and non-uniform upstream gradient")
```

`x=[-1, 0.25, 0.5, 0.75, 2]` 同时包含未激活（−1、0.25）、交界（0.5）和激活（0.75、2）三种位置，既检查了逐元素输出，也在编译路径上复查了阈值约定。这份检查只验证所列样本与指定 backend，不承诺任意版本、动态形状、张量子类或导出方式全部可用；`fullgraph=True` 也不意味着只执行一个 GPU 内核。非默认 stream 的一次通过，同样不是对所有并发组合的证明。

## 08｜比较性能：不能只拿 Triton 对 eager

为了延续第五篇的口径，至少比较三条路径：原生 eager、Triton 实现、编译后的原生函数。既然本篇强调 `triton_op` 对编译器可见，再补上一条**编译后的 Triton 包装函数**，就能把“替换执行路径”和“编译器介入后两种表达”这两个问题分开看：

| 比较 | 主要回答什么 |
| --- | --- |
| 普通调用 Triton 与原生 eager | 替换执行路径后的整体调用收益 |
| compiled Triton 与 compiled 原生 | 两种表达在编译环境中的表现 |
| Triton 与 CUDA C++ | 指定实现和配置下的性能差异 |

如果编译器已经很好地融合这条运算链，Triton 未必更快。PyTorch 调优指南本来就将逐元素融合列为编译优化的典型场景。[6] 完整跑分脚本见附录 A：它只测 float32 前向，输入已在 GPU，关闭梯度记录，每条路径返回新输出，预热与编译放在计时外。

需要提醒的是，这些比较仍包含包装、输出分配等调用成本，**不能据此直接给两种语言的内核性能排名**。若要真正报告跨实现差异，还应注明数值规则与编译设置的差别：第五篇的 CUDA C++ 实现显式处理乘加舍入，本篇则把浮点细节交给 Triton 编译器。原文采用容差比较正是为了容纳这类差异，而不是宣称逐位一致。

计时本身也包含 Python 调用、框架开销、输出分配和这一组 GPU 工作的完成等待，不是纯内核时间；100 次调用的组平均值也不是线上请求延迟的 P99。缓存复用、时钟、共享 GPU 负载都会影响结果，应记录环境并多次独立重复。

**本文没有实测结果表。**没有启用可选 CUDA C++ 路径，就不能从前三条路径的结果推断它与 CUDA C++ 谁更快。即使编译原生实现已经足够快，也是一项有用的结论。

## 09｜那么，C++ 和 CUDA 还需要学吗？

需要根据任务判断。先给出坐标系：Triton 可用于逐元素、归约和矩阵等计算，开源社区也有 FlashAttention 的 Triton 实现可查；而具体选择 Triton、CUDA C++ 还是成熟库，应根据目标硬件、计算模式、集成约束与实测结果判断。在 PyTorch 生态内，Triton 还有一个特殊身份——`torch.compile` 的默认后端 Inductor 会把适合的计算生成或编译为 Triton 内核再执行。[9][10] 这张表服务的是选型，不是排名。

| 任务或约束 | 优先评估的路线 |
| --- | --- |
| 原生表达式清楚，编译优化已满足目标 | PyTorch 原生算子与 torch.compile |
| 想研究可分块的逐元素、归约或矩阵类计算 | Triton，并按实际模式验证与调优 |
| 已有成熟 CUDA 库或 C++ 工程需要接入 | C++／CUDA 扩展与相应框架注册 |
| 依赖具体底层功能、特殊运行时或非 Python 部署约束 | 检查各工具支持范围，必要时使用 CUDA C++ |
| 目标是跨框架部署 | 分别设计适配、构建与验证，不凭语言名称推断兼容 |

这不是性能排名。Triton 减少了某些表达和工程成本，CUDA C++ 提供另一种控制与集成方式；两者的实际能力还取决于版本、后端与具体计算。

“代码更短”也不等于“优化已经完成”。数据布局、访存、寄存器压力、并行配置和归约方式，仍会影响运行效率。Triton 的自动调优工具可以帮助搜索配置，但需要合理候选与测量，不能把搜索成本或错误结果藏进最终加速比。进阶方向是把本篇的固定配置换成 `triton.autotune`，最小写法见附录 A。

## 10｜这个系列最终要学会的，是怎样交付一项计算

从平方到 PReLU，从 C++／CUDA 到 TensorFlow，再到 Triton，写法可以变化，几个问题始终存在：

1. 数学语义是什么，特殊位置采用什么约定？
2. 输入、输出、设备、布局和内存修改关系是什么？
3. 梯度怎样传递，怎样用独立方法核对？
4. 框架与编译器怎样认识这项计算？
5. 在真实热点和公平基线下，是否有稳定收益？

如果只是想更快完成这段运算，先运行原生实现与编译基线；如果要研究新的执行方式，再尝试 Triton 或 CUDA C++。选工具之前先明确目标，才能知道省下的是代码、开发时间，还是实际运行成本。

---

## 附录 A｜完整三文件工程、测试矩阵与跑分脚本

### 运行方式

本例需要兼容的 NVIDIA GPU、驱动、CUDA 版 PyTorch，以及提供 `triton_op`、`wrap_triton` 的相应版本组合（`triton_op` 自 PyTorch 2.6 起提供）。Triton 本身支持的平台与硬件应查看官方兼容说明；本篇没有验证其他 GPU 后端。[7]

建议按所选 PyTorch 的依赖约束安装匹配版本，不要为了获得某个新 API 就直接覆盖成任意最新版 Triton。

```bash
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.version.cuda); print(torch.cuda.is_available())"
python test_affine_triton.py
python bench_affine_triton.py
```

可选 CUDA C++ 对比：

```bash
WITH_CUDA_CPP=1 python bench_affine_triton.py
```

纯 Triton 示例没有要求你手写 `nvcc` 构建脚本，但不表示没有设备编译工具或兼容性依赖。启用第五篇的 CUDA C++ 路径时，则仍需其本地 Toolkit、C++ 编译器和 Ninja 环境。

**当前环境未执行以上代码。**文中的 PASS 是断言成立后程序才会打印的字符串，不是已有测试结论；首次 JIT、后端编译与缓存成本也没有被本文实测。实际运行后，再把版本记录与结果补进本节。

### test_affine_triton.py 与完整测试矩阵

下面各段按顺序组成 `test_affine_triton.py`，其中编译比对片段与正文第 07 节一致，此处并入完整文件。

#### 一：数值与输入梯度

```python
import torch
import triton
from affine_triton import affine_relu

if not torch.cuda.is_available():
    raise RuntimeError("需要可用的 CUDA GPU")
torch.manual_seed(0)
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(0))

def reference(x):
    return torch.relu(x * 2.0 - 1.0)

for dtype in (torch.float32, torch.float64):
    for shape in [(0,), (), (1,), (257,), (1023,), (1024,), (1025,), (2, 3, 5)]:
        x = (torch.rand(shape, device="cuda", dtype=dtype) * 4 - 2).requires_grad_(True)
        xr = x.detach().clone().requires_grad_(True)
        y, yr = affine_relu(x), reference(xr)
        g = torch.randn_like(y)
        gx, = torch.autograd.grad(y, (x,), grad_outputs=g)
        gr, = torch.autograd.grad(yr, (xr,), grad_outputs=g)
        torch.testing.assert_close(y, yr, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(gx, gr, rtol=1e-5, atol=1e-6)
        assert (y.shape, y.dtype, y.device) == (x.shape, x.dtype, x.device)

x = torch.tensor([-1., 0.25, 0.5, 0.75, 2.], device="cuda", requires_grad=True)
g = torch.tensor([1., 2., 3., 4., 5.], device="cuda")
gx, = torch.autograd.grad(affine_relu(x), (x,), grad_outputs=g)
torch.testing.assert_close(gx, torch.tensor([0., 0., 0., 8., 10.], device="cuda"))
print("PASS values, gradients, shape boundaries and threshold convention")
```

1023、1024、1025 分别检查不足一块、刚好一块和进入下一块；257 则保留与前面 CUDA 示例的联系。标量与空张量也没有被遗漏。

#### 二：数值梯度与注册交互

```python
x = torch.tensor([-0.7, 0.1, 0.9, 1.3], device="cuda",
                 dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(affine_relu, (x,),
                                eps=1e-6, atol=1e-5, rtol=1e-3)

op = torch.ops.triton_lab8.affine_relu.default
for need_grad in (False, True):
    x = torch.randn(257, device="cuda", requires_grad=need_grad)
    print("opcheck:", need_grad, torch.library.opcheck(op, (x,)))
print("PASS numerical gradient and registration checks")
```

数值差分避开 `x=0.5`，非光滑点单独检查约定。opcheck 检查注册交互，不替代梯度的数学验证。

#### 三：编译后的前向与一阶反向

```python
compiled_affine = torch.compile(affine_relu, backend="inductor",
                                fullgraph=True, dynamic=False)

x = torch.tensor([-1., 0.25, 0.5, 0.75, 2.], device="cuda", requires_grad=True)
xr = x.detach().clone().requires_grad_(True)
g = torch.tensor([1., 2., 3., 4., 5.], device="cuda")
y, yr = compiled_affine(x), reference(xr)
torch.testing.assert_close(y, yr, rtol=1e-5, atol=1e-6)
gy, = torch.autograd.grad(y, (x,), grad_outputs=g)
gr, = torch.autograd.grad(yr, (xr,), grad_outputs=g)
torch.testing.assert_close(gy, gr, rtol=1e-5, atol=1e-6)
print("PASS compiled full output and non-uniform upstream gradient")
```

这里直接比较输出张量，而不是先把输出压成平方和标量，能保留逐元素差异；固定输入覆盖激活、未激活与交界三种位置。

#### 四：非默认 stream 中的连续任务

```python
with torch.cuda.device(0):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = torch.empty(1025, device="cuda:0")
        x.fill_(0.75)
        y = affine_relu(x)
        z = y + 1.0
    stream.synchronize()
    torch.testing.assert_close(z, torch.full_like(z, 1.5))
print("PASS non-default stream smoke check")
```

保留 x、y、z 直到执行流完成；一次通过不是对所有并发组合的证明。非法输入、多卡交接、复杂布局和部署前的异常路径仍需继续验证。

### bench_affine_triton.py：跑分脚本

下面三段按顺序组成 `bench_affine_triton.py`。主实验只依赖本篇三个文件；若要把第五篇的 CUDA C++ 融合实现也纳入同一轮比较，可复制表中指定的三个旧文件，然后设置 `WITH_CUDA_CPP=1`。它们会使用第五篇的独立命名空间，不替换本篇 Triton 实现。

```python
import os
import random
import statistics
import time
import torch
import triton
from affine_triton import affine_relu

def eager(x):
    return torch.relu(x * 2.0 - 1.0)

def measure(fn, x):
    torch.cuda.synchronize(x.device)
    start = time.perf_counter()
    for _ in range(100):
        result = fn(x)
    torch.cuda.synchronize(x.device)
    return (time.perf_counter() - start) * 1000 / 100
```

```python
@torch.inference_mode()
def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("float32; BLOCK_SIZE=1024; num_warps=4; repeats=100; rounds=7")
    print("compile: inductor/default/fullgraph=True/dynamic=False")
    # 本脚本要求 compiled 基线可用；失败直接报错，不静默替换。
    compiled_eager = torch.compile(eager, backend="inductor", mode="default",
                                   fullgraph=True, dynamic=False)
    compiled_triton = torch.compile(affine_relu, backend="inductor",
                                    mode="default", fullgraph=True, dynamic=False)
    base_functions = {"eager": eager, "triton": affine_relu,
                      "compiled_eager": compiled_eager,
                      "compiled_triton": compiled_triton}
    if os.environ.get("WITH_CUDA_CPP") == "1":
        # 可选：把第五篇的 affine.py、affine_op.cpp、affine_kernel.cu 放在本目录。
        from affine import affine_relu as cuda_affine
        base_functions["cuda_cpp"] = lambda x: cuda_affine(x, 2.0, -1.0)

    for n in (1_024, 65_536, 1_000_000, 8_000_000):
        x = torch.rand(n, device="cuda", dtype=torch.float32) * 4.0 - 2.0
        for fn in base_functions.values():
            torch.testing.assert_close(fn(x), eager(x), rtol=1e-5, atol=1e-6)
            for _ in range(20):
                fn(x)
        torch.cuda.synchronize(x.device)
        samples = {name: [] for name in base_functions}
        for _ in range(7):
            order = list(base_functions)
            rng.shuffle(order)
            for name in order:
                samples[name].append(measure(base_functions[name], x))
        baseline = statistics.median(samples["eager"])
        print("\nN=", n)
        for name, values in samples.items():
            median = statistics.median(values)
            p25, _, p75 = statistics.quantiles(values, n=4, method="inclusive")
            print(f"{name:16s} median={median:.6f} ms "
                  f"p25={p25:.6f} p75={p75:.6f} "
                  f"eager/time={baseline / median:.3f}x")
```

```python
if __name__ == "__main__":
    main()
```

正式计时前每条路径各预热 20 次，旨在把 Triton 首次 JIT、Inductor 后端编译与缓存建立排除在计时窗口外。`compiled_triton` 与 `compiled_eager` 都走编译路径，用来回答“两种表达在编译环境中的表现”，仍不是对两种语言内核性能的最终判定。

### 自动调优的最小写法

把本篇的固定配置换成 `triton.autotune`，示意如下（不在本篇三文件工程中）：

```python
@triton.autotune(
    configs=[  # 候选配置组：编译器会逐个实测后按 benchmark 择优
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    ],
    key=["n"],  # n 是元素数量；其取值变化会参与调优配置选择
)
@triton.jit
def affine_relu_kernel_tuned(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    # 内核体与本篇 affine_relu_kernel 完全相同
    ...
```

首次遇到新的 `n` 会触发一轮候选实测，之后按缓存直接选用；这正是正文说的“搜索成本”，要与稳态性能分开看待。需要记住，这组候选只是在给定集合中择优，不代表找到了全局最优配置。

---

## 参考资料

核对日期：2026 年 10 月 19 日。在线文档可能更新，请匹配本地安装版本。

1. [Triton — Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)：program、逻辑块、索引、mask 与内核启动。
2. [Triton — triton.jit](https://triton-lang.org/main/python-api/generated/triton.jit.html)：JIT 内核接口与编译机制。
3. [Triton — triton.language](https://triton-lang.org/main/python-api/triton.language.html)：`program_id`、`arange`、`load`、`store` 等语言操作。
4. [Using User-Defined Triton Kernels with torch.compile](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)：直接调用与结构化包装、triton_op、wrap_triton 和自动微分。
5. [torch.library](https://docs.pytorch.org/docs/stable/library.html)：triton_op 的可追踪实现、修改约定与相关注册接口。
6. [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-operations)：逐元素融合与编译优化。
7. [Triton — Installation](https://triton-lang.org/main/getting-started/installation.html)：安装及平台／硬件兼容信息入口。
8. [Triton — Config](https://triton-lang.org/main/python-api/generated/triton.Config.html)：num_warps、编译配置与自动调优候选的含义。
9. [Triton: A Language and Compiler for GPU Computing](https://dl.acm.org/doi/10.1145/3315508.3329973)（PPoPP 2019）与 [FlashAttention 仓库的 Triton 实现](https://github.com/Dao-AILab/flash-attention)：Triton 的语言与编译器定位及 attention 类内核实践示例。
10. [PyTorch — torch.compiler Inductor](https://docs.pytorch.org/docs/stable/torch.compiler_inductor.html)：Inductor 后端及其与 Triton 的关系。