# 单独运行没问题，为什么一开编译就出错？

> **自定义算子 · 第六篇**
>
> 自定义 CUDA 算子已经能运行，也接上了梯度，为什么放进 `torch.compile` 仍可能失败？编译器在真正计算之前，还需要知道输出“长什么样”。这一次，我们补齐这份描述，并把不同层次的验证分开。
>
> 本文面向具备基础 Python 知识的 AI 兴趣者，提供独立四文件工程。当前创作环境未安装 PyTorch，未执行 C++／CUDA 编译、opcheck、gradcheck 或图编译测试；文中代码给出验证方法，不虚构通过日志或性能结果。资料核对日期：2026 年 10 月 19 日。

<!-- more -->

## 01｜框架会执行它，编译器为什么还要问？

设想你在优化摄像头模型的一段计算。自定义激活函数已经完成两项检查：输入进去，输出正确；损失反向传播，输入和参数也能得到梯度。

接着，把包含它的函数交给 `torch.compile`。问题可能出现在真正执行 CUDA 计算之前：框架无法为这项外部计算推导输出信息。

这不一定说明线程索引写错了。运行时与编译分析阶段，面对的不是同一项任务。

- 运行时拿着真实数据，要求把输出算出来。
- 编译分析阶段需要理解后续计算如何连接，往往先推导张量的形状、类型、设备等信息，而不读取真实元素值。

PyTorch 为这类分析使用 FakeTensor。官方自定义算子教程要求提供相应的 fake 实现，以便根据输入元数据描述输出元数据。[1]

**前向能运行、梯度公式正确、编译路径可用，是相关但不同的三项能力。**本篇从已经接好梯度的 PReLU 出发，补上第三项。

## 02｜没有真实像素，也能知道输出有多大

继续使用共享一个参数的 PReLU：

$$
y_i=\begin{cases}x_i,&x_i>0\\a x_i,&x_i\le0\end{cases}
$$

输入是一张形状为 `(2, 3, 5)` 的特征张量，输出仍是 `(2, 3, 5)`。即使不知道任何元素数值，也可以确定这一点。

如果输入是 float32、位于 CUDA 设备 0，输出也应是对应 dtype 和设备；本实现返回新分配、与输入形状相同的连续结果。

这些信息属于元数据：

| 信息 | 编译分析为什么需要它 |
| --- | --- |
| shape | 后续操作接收什么维度和大小 |
| dtype | 使用什么数据类型与运算规则 |
| device | 计算和数据属于哪个设备 |
| stride／layout | 逻辑元素怎样对应存储布局 |
| 别名与修改约定 | 输出是否共享输入存储，输入是否会改变 |

最后一项也依赖 schema 等接口契约，不是仅靠返回一个形状正确的张量就能交代清楚。

FakeTensor 可以理解为保留这些特征、但不承载真实元素数据的张量。它能表现出 CUDA 设备信息，并不等于为了分析就分配一份真实的 GPU 输入。[2] FakeTensor 与 meta 张量有关联，但并不等同：FakeTensor 会保留 CUDA 这类逻辑设备信息，所以当报错里出现 `Meta` 时，不能简单理解为计算被搬到了另一块设备。

所以 fake 实现不会再次完成 PReLU 数值计算。它提供的是：“满足这些输入条件时，输出应具有什么属性。”

整个流程可以概括成一条链：

> fake 输入 → 元数据规则 → fake 输出 → 后续形状推导

也可以对照两个阶段各自在做什么：

| 阶段 | 拿到什么 | 做什么 |
| --- | --- | --- |
| 编译分析 | 带元数据的 fake 输入 | 按规则推导输出的 shape／dtype／device／stride |
| 真实执行 | 真实数据 | 执行内核，算出真实元素值 |

fake 实现是这条链上的“中转站”：在这个分析阶段，编译器依据元数据推导输出属性，不读取输入的真实元素值，却能凭这些属性把后续节点串成完整的图。少任一环（比如 dtype 描述错了），图就在错误前提上被拼出来。

## 03｜为什么返回 `empty_like(input)` 也需要认真设计？

对本例，fake 实现的核心确实可以很短：

```python
return torch.empty_like(input)
```

但短不代表可以照抄到所有算子。它之所以成立，是因为真实前向也使用 `empty_like` 分配输出，且接口限制输入为标准连续布局。

几个常见错误是：

- 用 `torch.empty(input.shape)`，漏掉 dtype 和设备，导致输出描述与真实结果不一致。
- 只顾 shape，不考虑转置、切片或特殊布局下的 stride。
- 返回 `input` 本身，意外宣称输出与输入共享存储。
- 使用 `input.data_ptr()`、`.item()` 或 `.cpu().numpy()`，试图读取本来就不存在的真实数据。

官方文档强调，fake 实现应由可分析的 PyTorch 操作组成，不能直接访问张量的真实存储；输出元数据需要匹配实际实现。[2][3]

本例也会把输入约束写进 fake 实现：真实内核只支持同设备、同 dtype、连续张量，fake 路径不能假装所有输入都支持。否则可能发生“编译分析通过，运行时才发现不合法”。

### 如果输出长度取决于数值呢？

例如筛选所有大于阈值的元素，仅知道输入长度，并不能知道最终保留几个。此时不能偷看输入值，也不能随便拿输入长度冒充输出长度。

这类操作涉及数据依赖形状及相应的动态尺寸机制，官方文档有专门接口。[2] 本篇选择形状不依赖数值的 PReLU，先讲清普通路径，不把复杂动态输出混进来。

## 04｜四种验证，不要相互替代

本篇会用到不同检查，它们回答的问题不同：

| 检查 | 主要回答什么 | 不能据此宣布什么 |
| --- | --- | --- |
| 数值与原生参照比较 | 输出和梯度是否符合所测语义 | 所有注册规则都正确 |
| `gradcheck` | 解析梯度是否与数值差分相符 | 编译路径一定可用 |
| `opcheck` | 所测样本下，注册和相关框架交互是否一致 | 梯度在数学上一定正确 |
| 编译路径对照 | 指定 backend 下的前向结果和反向梯度是否一致 | 任意后端、形状和版本都支持，也不证明加速 |

`opcheck` 的文档列出了 schema、自动微分注册、FakeTensor 和 AOT 分派等检查；其范围与默认项可能随版本变化，应查看实际安装版本。[2]

尤其要注意：如果 eager 与编译路径都使用同一个错误的梯度公式，两者仍可能一致。因此独立数学参照和 `gradcheck` 不能省略。[1]

## 05｜工程安排：计算不变，新增 fake 实现

本篇继续采用 CUDA 前向、PyTorch 原生算子表达反向的方式。输入 x、参数 a 必须在同一 CUDA 设备，dtype 相同且为 float32 或 float64，标准连续稠密布局，a 固定为 `(1,)`；x 可为空、标量或多维，元素数不超过 2³¹−1。

输出新分配，不修改输入。零点反向沿用第四篇的约定：输入梯度取 `a × grad_output`，参数梯度的该位置贡献为零。

| 文件 | 内容 |
| --- | --- |
| `prelu_op.cpp` | schema 和 CUDA 实现注册，完整代码见附录 A |
| `prelu_kernel.cu` | 前向、校验、设备与 stream 管理，完整代码见附录 B |
| `prelu.py` | 动态库加载、fake 实现、自动微分注册 |
| `test_prelu.py` | 独立数值、梯度、注册与编译验证 |

新命名空间使用 `prelu_lab6`，避免与第四篇的定义混淆。在一个新目录中创建这四个文件即可，不依赖旧工程的编译产物。

## 06｜先加载定义，再注册 fake 与反向

下面三个代码块按顺序组成完整的 `prelu.py`。

### 先让定义存在，fake 注册才有对象

```python
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

if not torch.cuda.is_available():
    raise RuntimeError("本工程需要可用的 CUDA GPU")
for api in ("register_fake", "register_autograd", "opcheck"):
    if not hasattr(torch.library, api):
        raise RuntimeError(f"当前 PyTorch 缺少 torch.library.{api}")

root = Path(__file__).resolve().parent
load(
    name="prelu_lab6_ext",
    sources=[str(root / "prelu_op.cpp"), str(root / "prelu_kernel.cu")],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2"],
    is_python_module=False,
    verbose=True,
)
```

加载时执行 C++ 注册逻辑，之后才可以为 `prelu_lab6::prelu` 增加其他注册。官方教程明确提醒混合 C++／Python 注册的加载顺序。[1]

### 再补 fake 实现，只描述输出元数据

```python
@torch.library.register_fake("prelu_lab6::prelu")
def prelu_fake(input, a):
    torch._check(input.device.type == "cuda", lambda: "input must be CUDA")
    torch._check(input.device == a.device, lambda: "inputs must share device")
    torch._check(input.layout == torch.strided, lambda: "use strided layout")
    torch._check(a.layout == torch.strided, lambda: "use strided layout")
    torch._check(input.is_contiguous(), lambda: "input must be contiguous")
    torch._check(a.is_contiguous(), lambda: "a must be contiguous")
    torch._check(a.dim() == 1, lambda: "a must have rank 1")
    torch._check(a.numel() == 1, lambda: "a must have one element")
    torch._check(input.dtype == a.dtype, lambda: "dtype must match")
    torch._check(input.dtype in (torch.float32, torch.float64),
                 lambda: "use float32 or float64")
    torch._check(input.numel() <= 2**31 - 1, lambda: "too many elements")
    return torch.empty_like(input)
```

`torch._check` 是 PyTorch 在这类实现中使用的条件检查接口，可表达包括符号尺寸在内的约束。[2] 尽量不要为了检查形状而先把符号尺寸转成普通整数，那可能增加不必要的具体化和限制。

这 12 条 `torch._check` 与附录 B 中 `prelu_kernel.cu` 的 `TORCH_CHECK` 覆盖同一组约束（C++ 侧把部分条件合并在一条里）。fake 与真实实现维护相同的输入约束：在本工程声明支持范围内的具体输入上，fake 侧放行，真实前向也应放行，反之亦然。修改支持范围时，两处检查和对应测试需要同步更新，否则会出现“分析通过、运行时报错”的缺口。（符号尺寸、专门化与 guards 的处理可能不同，这里不对它们作等同承诺。）

本例分析的是保留 CUDA 设备属性的 fake 输入；这段检查没有承诺支持任意独立 meta／CPU 输入。

注意，我们没有读 x 的数值，也没有执行 CUDA 内核。`empty_like` 在这里描述输出，不生成一份真实 PReLU 结果。

### 反向公式独立保留，梯度不交给 fake

```python
def setup_context(ctx, inputs, output):
    input, a = inputs
    ctx.save_for_backward(input, a)

def backward(ctx, grad_output):
    input, a = ctx.saved_tensors
    positive = input > 0
    grad_input, grad_a = None, None
    if ctx.needs_input_grad[0]:
        slope = torch.where(positive, torch.ones_like(input), a.reshape(()))
        grad_input = grad_output * slope
    if ctx.needs_input_grad[1]:
        values = torch.where(positive, torch.zeros_like(input), grad_output * input)
        grad_a = values.sum().reshape_as(a)
    return grad_input, grad_a

torch.library.register_autograd(
    "prelu_lab6::prelu", backward, setup_context=setup_context
)
prelu = torch.ops.prelu_lab6.prelu.default
```

反向梯度分别是输入分支的局部斜率乘以上游梯度，以及所有负分支参数贡献之和。`a.reshape(())` 避免标量输入的梯度多出一个维度，归约结果则恢复到参数的 `(1,)`。

fake 实现与反向公式各司其职：前者描述结果的元数据，后者计算真实梯度，不能相互替代。

## 07｜完整测试：先验证数学，再检查接线

下面五个代码块按顺序组成 `test_prelu.py`。先看每个块回答什么问题，再读代码：

| 代码块 | 回答什么问题 |
| --- | --- |
| 一：先用独立参照钉死语义 | 前向与梯度是否等于独立参照实现，零点行为是否符合约定 |
| 二：再用数值差分校验梯度 | 解析梯度是否与数值差分相符 |
| 三：opcheck 默认注册检查 | 安装版本默认的注册、fake、分派检查项是否通过 |
| 四：aot_eager 与 Inductor 分层排查 | 两条编译路径下，完整输出、标量损失与一阶梯度是否一致 |
| 五：非法输入拒绝冒烟 | 不支持的输入是否被响亮拒绝，而不是静默算错 |

### 一：先用独立参照钉死语义，含零点约定

```python
import torch
from prelu import prelu

torch.manual_seed(0)
print("PyTorch:", torch.__version__,
      "PyTorch built with CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

def reference(x, a):
    return torch.where(x > 0, x, a.reshape(()) * x)

def compare_eager(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype).requires_grad_(True)
    a = torch.tensor([0.25], device="cuda", dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    ar = a.detach().clone().requires_grad_(True)
    y, yr = prelu(x, a), reference(xr, ar)
    g = torch.randn_like(y)
    grads = torch.autograd.grad(y, (x, a), grad_outputs=g)
    refs = torch.autograd.grad(yr, (xr, ar), grad_outputs=g)
    torch.testing.assert_close(y, yr)
    for actual, target in zip(grads, refs):
        torch.testing.assert_close(actual, target)

for dtype in (torch.float32, torch.float64):
    for shape in [(0,), (), (1,), (257,), (2, 3, 5)]:
        compare_eager(shape, dtype)

x = torch.tensor([-2., 0., 3.], device="cuda", requires_grad=True)
a = torch.tensor([0.25], device="cuda", requires_grad=True)
gx, ga = torch.autograd.grad(prelu(x, a).sum(), (x, a))
torch.testing.assert_close(gx, torch.tensor([0.25, 0.25, 1.], device="cuda"))
torch.testing.assert_close(ga, torch.tensor([-2.], device="cuda"))
print("PASS eager reference and zero-point convention")
```

第四篇已经演示过与原生 PReLU 在非零输入上的独立对照；这里保留同语义的原生组合参照，并继续检查约定的零点行为。

### 二：再用数值差分校验解析梯度

```python
x = torch.tensor([-1.5, -0.7, 0.4, 2.0], device="cuda",
                 dtype=torch.float64, requires_grad=True)
a = torch.tensor([0.25], device="cuda", dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(prelu, (x, a),
                                eps=1e-6, atol=1e-5, rtol=1e-3)
print("PASS gradcheck")
```

float64 减少有限差分中的精度问题，输入避开不可微的分支交界。这个检查仍然使用真实自定义前向，不是只检查另写的参考函数。数值差分原理见官方 gradcheck 说明。[4]

### 三：注册完整性交给 opcheck 默认检查集

```python
samples = [
    ((257,), torch.float32, True, True),
    ((2, 3, 5), torch.float32, True, False),
    ((), torch.float64, False, True),
    ((0,), torch.float32, True, True),
    ((257,), torch.float64, False, False),
]
for shape, dtype, need_x, need_a in samples:
    x = torch.randn(shape, device="cuda", dtype=dtype).requires_grad_(need_x)
    a = torch.tensor([0.25], device="cuda", dtype=dtype, requires_grad=need_a)
    result = torch.library.opcheck(prelu, (x, a))
    print("opcheck:", shape, dtype, need_x, need_a, result)
```

不要把 `test_utils` 缩减到只剩容易通过的项目后，仍称“完整 opcheck 通过”。本例调用安装版本的默认检查集，遇到失败应保留具体检查名与异常。

其中与 AOT 分派有关的检查会比较相关执行路径，但不等于对所有动态图、设备和导出方式作出支持承诺。

### 四：分层排查，先 aot_eager 图捕获再 Inductor 代码生成

```python
def objective(x, a):
    return prelu(x, a).square().sum()

def reference_objective(x, a):
    return reference(x, a).square().sum()

compile_cases = [((257,), torch.float32),
                 ((2, 3, 5), torch.float32),
                 ((), torch.float64)]
for backend in ("aot_eager", "inductor"):
    compiled = torch.compile(objective, backend=backend,
                             fullgraph=True, dynamic=False)
    for shape, dtype in compile_cases:
        x = torch.randn(shape, device="cuda", dtype=dtype).requires_grad_(True)
        a = torch.tensor([0.25], device="cuda", dtype=dtype, requires_grad=True)
        xr = x.detach().clone().requires_grad_(True)
        ar = a.detach().clone().requires_grad_(True)
        loss, expected = compiled(x, a), reference_objective(xr, ar)
        grads = torch.autograd.grad(loss, (x, a))
        refs = torch.autograd.grad(expected, (xr, ar))
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
        for actual, target in zip(grads, refs):
            torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-6)
        print("PASS compiled forward and first-order gradients:", backend, shape)
```

这两条路径用于逐步缩小问题范围：`aot_eager` 可以帮助排查图捕获及自动微分处理，`inductor` 再检查所用代码生成后端的路径。它们都不是“绕过注册错误”的替代方法。

`fullgraph=True` 要求被捕获的前向形成单个 FX 图，否则报错；它不等于整个程序只有一个 GPU 内核，也不表示优化器、反向执行和所有 Python 代码都被合成同一张图。[5]

`dynamic=False` 是本篇的明确测试配置，不同形状可能触发不同专门化或重编译。即使默认 opcheck 的某些项目涉及符号尺寸，也不能据此宣称任意动态形状组合都已验证。

这里实际调用 `autograd.grad`，才能触发并检查相关反向执行；只打印前向 loss，不能证明编译后训练路径正确。

不过，标量损失一致，还不能单独证明整个输出张量一致——`square().sum()` 会丢掉部分符号信息。再补一项更直接的对照：编译返回 `prelu(x, a)` 本身的函数，比较完整输出张量，并用非全 1 的 `grad_outputs` 比较输入梯度与参数梯度。

```python
def forward_only(x, a):
    return prelu(x, a)

for backend in ("aot_eager", "inductor"):
    compiled_fwd = torch.compile(forward_only, backend=backend,
                                 fullgraph=True, dynamic=False)
    for shape, dtype in compile_cases:
        x = torch.randn(shape, device="cuda", dtype=dtype).requires_grad_(True)
        a = torch.tensor([0.25], device="cuda", dtype=dtype, requires_grad=True)
        xr = x.detach().clone().requires_grad_(True)
        ar = a.detach().clone().requires_grad_(True)
        out, ref = compiled_fwd(x, a), reference(xr, ar)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)  # 完整输出一致
        g = torch.randn_like(out) * 3.0 + 0.5  # 非全 1 的上游梯度
        grads = torch.autograd.grad(out, (x, a), grad_outputs=g)
        refs = torch.autograd.grad(ref, (xr, ar), grad_outputs=g)
        for actual, target in zip(grads, refs):
            torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-6)
```

验证层次因此分成三级：完整输出一致 → 指定上游梯度下一致 → 包含后续运算的训练目标一致。这一块补的是第一级，上一块补的是第三级。

注意，空张量目前只进入了 eager 参照与 `opcheck`，未进入上面的显式 backend 对照；如需覆盖，可单独补一例，不必扩成庞大的测试矩阵。

### 五：最后确认，非法输入要被响亮拒绝

```python
def expect_rejection(x, a):
    try:
        prelu(x, a)
    except (RuntimeError, NotImplementedError):
        return
    raise AssertionError("非法输入本应被拒绝")

expect_rejection(torch.ones(2, 3, device="cuda").t(),
                 torch.ones(1, device="cuda"))
expect_rejection(torch.ones(4, device="cuda", dtype=torch.float32),
                 torch.ones(1, device="cuda", dtype=torch.float64))
expect_rejection(torch.ones(4, device="cuda"), torch.ones(2, device="cuda"))
torch.cuda.synchronize()
print("PASS rejected-input smoke checks")
```

这些是拒绝路径的冒烟检查，不能证明任意异常都来自预期约束。若用于正式工程，应继续核对错误内容，并按第三篇覆盖多设备、stream 与内存生命周期场景。

## 08｜运行时，请保留真实的失败信息

环境需要支持 `register_fake`、`register_autograd` 与 `opcheck` 的 PyTorch、兼容的 NVIDIA 驱动、本地 CUDA Toolkit、C++ 编译器、Ninja，以及该平台上的 Inductor GPU 编译依赖。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
c++ --version
python -m pip install ninja
python test_prelu.py
```

从首次构建到图编译可能耗时较长。本篇没有计时；需要性能数据时，应沿第五篇的方法单独预热和测量。工具链加载流程沿用官方扩展构建方式。[6]

**本文没有运行以上工程。**前述 `PASS` 字符串只是代码在断言成立后会打印的内容，不是本文已经获得的结果。实际运行应分别记录版本、backend、样本与失败阶段。

尤其不要用全局忽略编译错误、关闭检查或静默退回 eager，来把“能继续执行”包装成“编译接入已完成”。

### 记录一次运行，方便跨版本对照

不同 PyTorch 版本对 stride 元数据不一致等问题的检查覆盖可能不同，因此“默认检查集通过”只对特定版本、特定样本成立。实际运行时按下表记录，便于日后对照：

| 项目 | 记录内容 |
| --- | --- |
| 环境 | PyTorch、GPU、驱动、CUDA Toolkit 版本 |
| 注册检查 | `opcheck` 默认各项结果 |
| 编译路径 | `aot_eager`、Inductor 各自结果 |
| 覆盖范围 | dtype、形状、梯度需求 |
| 失败信息 | 失败阶段、检查名与完整异常 |

## 09｜怎样识别几类常见失败

排障的第一步是把失败归类。下面列出两类常见形态，再给一份对照表。这里是为等价形态改写，不是实测截图；实际措辞以安装版本为准，完整实测文本待真实运行后回填。

### 形态一：分析阶段找不到 fake 实现

```text
NotImplementedError: Could not run 'prelu_lab6::prelu' with arguments from
the 'Meta' backend because there are no Meta kernel(s) registered for
this function.
```

逐行拆解这份报错：

- `NotImplementedError`：分派层没找到实现——卡在编译分析阶段，与 CUDA 内核算得对不对无关；
- `'prelu_lab6::prelu'`：出问题的算子全名，直接指向本篇的命名空间，一眼定位漏注册的是谁；
- `'Meta'` backend：只带元数据的执行路径，正是 §01 说的“不读真实值”的那条路；
- `no ... kernel(s) registered`：根因——附录 A 注册了 CUDA 实现，却没按 §06 注册 fake 实现；
- 部分版本还会追加指向 `torch.library.register_fake` 的补充提示，具体措辞随版本变化。

### 形态二：类型或数值对照不一致

```text
RuntimeError: expected scalar type Float but found Double
```

或在正确性对照测试里表现为：

```text
torch.testing.assert_close failed: Tensor-likes are not close!
```

类型不匹配或数值对照失败，只说明某项预期没有满足，**不能直接归因于 fake 描述错误**。它可能来自输入类型不一致、其他算子的类型约束，也可能来自 fake 实现本身。正确做法是先定位失败阶段，再逐项检查实际与预期的 shape、dtype、device、stride 和数值，最后判断是否由 fake 实现引起。

顺带一提，`opcheck` 的 FakeTensor 检查会主动比较真实输出与 fake 输出的元数据，因此部分描述错误可以在编译产物运行之前就被发现，而不必等到首次执行。

### 对照表：先问它属于哪一类

| 现象或失败项目 | 优先检查 |
| --- | --- |
| 缺少 fake／meta 实现 | 是否在算子定义加载后完成注册 |
| fake 输出与实际输出不一致 | shape、stride、dtype、device、别名等契约 |
| 分析阶段出现数据访问错误 | fake 中是否调用了 `.item()`、指针或 NumPy |
| `test_schema` 失败 | 输入是否被修改，输出是否共享了未声明的存储 |
| `test_autograd_registration` 失败 | 是否正确注册反向，返回梯度数量与位置是否一致 |
| `gradcheck` 失败 | 公式、dtype、容差、非光滑点及数值条件 |
| aot_eager 失败 | 图捕获、fake、反向追踪及相应约束 |
| 只有 Inductor 失败 | 进一步检查后端支持、工具链与版本，不能直接断定内核错误 |

这是一份排查顺序，不是错误原因的一一对应表。真实异常还要结合最小复现与完整栈信息判断。

## 10｜一个可控的最小实验：改一处，看影响哪一步

与其只读现成的错误文本，不如自己制造一次失败，观察它出现在哪一步。下面是三组**实验任务**（需要真实运行后记录，本文未执行）：

| 操作 | 观察任务 |
| --- | --- |
| 暂时取消 fake 注册 | 记录编译或注册检查在哪一步失败、异常停在哪一阶段 |
| 恢复正确的 fake 实现 | 检查相同调用是否继续完成 |
| 在独立实验中故意返回错误 dtype 的 fake 输出 | 观察 `opcheck` 是否报告元数据不匹配 |

这样就能把“改了什么—影响了什么”连起来：漏注册通常停在分析阶段，表现为找不到 Meta 实现；元数据写错，则往往由 `opcheck` 的元数据对照先报出来。两组现象对应的失败阶段不同，正是区分“没接上”和“接错了”的依据。

## 11｜支持编译之后，CUDA 内核会自动被看穿吗？

不会。

对于本篇按普通自定义算子注册的外部 CUDA 计算，fake 实现让编译器知道输入输出属性，但没有把 CUDA 源码翻译成它可以任意改写的逐元素表达式。[3]

编译器可以处理周围的原生运算，并在运行时调用这个外部算子；不能因为“进入了图”，就断言自定义内核一定与周围运算融合，或比第五篇的原生编译路径更快。

截至这里，我们建立的是逐层接入的方法：

- 算术与边界，靠数值及输入测试验证。
- 训练规则，靠反向注册、参照和数值梯度检查验证。
- 编译交互，靠 fake 实现、opcheck 和具体 backend 路径验证。
- 性能收益，仍要回到统一口径的测量。

后续扩展篇会分别讨论 TensorFlow 适配与 Triton 路线。迁移语言或框架时，也应继续保持这些契约与验证习惯，而不只追求“示例成功输出一次”。

---

## 附录 A｜`prelu_op.cpp` 完整内容

```cpp
#include <ATen/ATen.h>
#include <torch/library.h>

at::Tensor prelu_cuda(const at::Tensor& input, const at::Tensor& a);

TORCH_LIBRARY(prelu_lab6, m) {
    m.def("prelu(Tensor input, Tensor a) -> Tensor");
}

TORCH_LIBRARY_IMPL(prelu_lab6, CUDA, m) {
    m.impl("prelu", &prelu_cuda);
}
```

## 附录 B｜`prelu_kernel.cu` 完整内容

分成两个代码块便于阅读，保存时按顺序放入同一个文件。

```cpp
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

template <typename scalar_t>
__global__ void prelu_kernel(const scalar_t* input, const scalar_t* a,
                             scalar_t* output, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x)
                      * blockDim.x + threadIdx.x;
    if (i < n) {
        const scalar_t value = input[i];
        output[i] = value > scalar_t(0) ? value : a[0] * value;
    }
}
```

```cpp
at::Tensor prelu_cuda(const at::Tensor& input, const at::Tensor& a) {
    TORCH_CHECK(input.is_cuda() && a.is_cuda(), "both inputs must be CUDA");
    TORCH_CHECK(input.device() == a.device(), "inputs must share device");
    TORCH_CHECK(input.layout() == c10::kStrided && a.layout() == c10::kStrided,
                "inputs must have strided layout");
    TORCH_CHECK(input.is_contiguous() && a.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(a.dim() == 1 && a.numel() == 1, "a must have shape (1,)");
    TORCH_CHECK(input.scalar_type() == a.scalar_type(), "dtype must match");
    TORCH_CHECK(input.scalar_type() == at::kFloat ||
                input.scalar_type() == at::kDouble, "use float32 or float64");
    TORCH_CHECK(input.numel() <= std::numeric_limits<int>::max(),
                "too many elements for this tutorial");

    const c10::cuda::CUDAGuard guard(input.device());
    auto output = at::empty_like(input);
    const int64_t n = input.numel();
    if (n == 0) return output;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "prelu_lab6_forward", [&] {
        prelu_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            input.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
```

## 参考资料

核对日期：2026 年 10 月 19 日。在线文档会更新，应匹配实际安装版本。本工程使用常规 ATen／LibTorch 接口，不作跨版本二进制兼容承诺。

1. [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)：C++／Python 混合注册顺序、FakeTensor、自动微分与算子测试。
2. [torch.library](https://docs.pytorch.org/docs/stable/library.html)：`register_fake`、`register_autograd`、`opcheck` 与数据依赖形状接口。
3. [Functional Python Custom Operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops_functional.html)：输出元数据匹配、fake 实现约束及普通自定义算子的编译边界。
4. [Gradcheck mechanics](https://docs.pytorch.org/docs/stable/notes/gradcheck.html)：数值差分与解析梯度的比较原理。
5. [Use fullgraph=True to Identify and Eliminate Graph Breaks](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html)：完整图捕获与图中断的含义。
6. [torch.utils.cpp_extension](https://docs.pytorch.org/docs/stable/cpp_extension.html)：动态扩展构建与加载。
