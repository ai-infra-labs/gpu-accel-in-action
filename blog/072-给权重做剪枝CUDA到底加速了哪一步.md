# 给权重做剪枝，CUDA 到底加速了哪一步？

> **TensorRT 推理加速 · 第 6 篇**
>
> 权重变成零，需要有人执行筛选。筛选做得快，与剪枝后的模型推理更快，是两件需要分别验证的事。本篇从一段“名字里有 CUDA，计算却回到 CPU”的代码出发，先在同一个框架里测出这段往返的代价，再写清 GPU 如何并行处理权重，以及怎样设计可比较的实验。
>
> 资料核对日期：2026 年 10 月 21 日。本文运行了 NumPy 阈值示例；两段 GPU 代码（PyTorch 对照实验与 PyCUDA 实验）仅通过 Python 语法检查，未在 NVIDIA GPU 上执行，因此不提供实测加速倍数。

## 01｜模型在 GPU，筛选为什么还在 CPU？

上一期，我们讨论了 INT8：用较低精度表示数字，再检查误差与部署收益。

这一期换一个思路：如果模型里有些权重可以不保留，能不能先把它们置零？

继续质检部署的场景。你准备在生成新引擎之前处理模型权重，写了一个叫 `cuda_prune` 的函数。模型已经放到 GPU，也导入了 PyCUDA，看起来很像 GPU 加速代码。

但一段常见的原始写法是这样的：

```python
weights = param.data.cpu().numpy()
mask = np.abs(weights) > threshold
weights[~mask] = 0
param.data = torch.from_numpy(weights).cuda()
```

沿着数据走一遍，就能看清实际发生了什么：GPU 权重先复制到 CPU，NumPy 在 CPU 上计算绝对值、比较阈值、赋零，随后结果再复制回 GPU。

**核心筛选没有在 GPU 上执行。** 将模型移到 GPU，或者导入 CUDA 库，都不能改变这几行 NumPy 操作的执行位置。

这段原始代码还直接操作 `.data`；框架写法里会换成明确的无梯度参数修改，后面那组对照实验则把这几行原样搬回来，作为被比较的一方。我们先不讨论这个模型能不能因此更快，先把“筛选在哪里完成”讲清楚。

## 02｜先定义规则：等于阈值，要不要保留？

本文采用一个简单的幅值阈值规则：

```text
|w| > t：保留原值
|w| ≤ t：设为 0
```

`w` 是一个权重，`t` 是非负阈值。注意判断的是绝对值，负的大权重不会因为“小于正阈值”而全部被删除；等于阈值的位置也要置零。

用一个教学数组检查：

| 原权重 | -0.5 | -0.125 | 0 | 0.125 | 0.5 |
|---|---:|---:|---:|---:|---:|
| 阈值 `t=0.125` 后 | -0.5 | 0 | 0 | 0 | 0.5 |

对应的 CPU 参考实现很短：

```python
import numpy as np

threshold = np.float32(0.125)
weights = np.array([-0.5, -0.125, 0, 0.125, 0.5], dtype=np.float32)
result = np.where(np.abs(weights) > threshold, weights, np.float32(0))
print(result)
```

这段 NumPy 示例已运行，数值结果与表格一致；数组的形状和字节数没有因置零而缩小。

为什么用 0.125？它只是一个能被二进制浮点数精确表示、便于核对边界的教学阈值，**不是模型剪枝的推荐值**。

固定阈值也不等于固定剪枝比例。不同层的权重分布不同，同一个阈值可能让某层几乎不变，另一层大部分变成零。幅值较小，更不意味着这个权重在任务中一定不重要。

我们现在定义的是一个可验证的筛选操作，还没有得到一个经过质量验证的模型压缩方案。

## 03｜先用 GPU 张量运算，不必立即手写 kernel

如果模型已经在 PyTorch 的 CUDA 设备上，直接使用 GPU 张量运算，就能避免刚才的 CPU 往返：

```python
import torch

def apply_threshold(model, threshold=0.125):
    # 前提：权重是有限数值，阈值为非负有限值；本例不处理参数化权重。
    if not (0 <= threshold < float("inf")):
        raise ValueError("阈值必须是非负有限数。")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("weight"):
                if not param.is_cuda:
                    raise ValueError("请先把模型移到目标 CUDA 设备。")
                param.masked_fill_(param.abs() <= threshold, 0)
```

`abs`、比较和 `masked_fill_` 针对 CUDA 张量执行，因而相关计算在 GPU 上进行。`masked_fill_` 根据布尔掩码原地填值；`no_grad` 表明这次参数修改不记录自动求导图。[1]

这段代码适合说明“怎样避免回 CPU”。它没有自定义 CUDA 源码，也不能据此推断框架一定启动了多少个**内核（kernel）**。

示例里用 `name.endswith("weight")` 挑参数，是为了让演示代码尽量短。**这个条件只在本例成立**：演示网络是三个普通线性层，`endswith("weight")` 恰好只命中这三个权重张量。换成真实模型，归一化层的缩放参数、嵌入层的权重同样满足这个后缀，而它们的形状、量纲和“小权重不重要”的前提完全不同。移植这段代码时，应显式列出要处理的层（例如只处理 `nn.Linear`），而不是仅凭参数名批量剪枝。

这里还有两个边界：

- 这是一次性参数修改。继续训练时，零权重可能再次更新；若要持续保持稀疏约束，需要额外机制。
- PyTorch 的剪枝工具还提供权重与 mask 的重参数化管理。本文的直接赋零不等同于完整的剪枝训练流程。[2]

接下来手写 kernel，是为了看清这项操作怎样映射到线程，不是因为框架张量操作不算 GPU 计算。

## 04｜把那段原始写法换掉，省下来的时间在哪里？

上一节给出了框架写法，但它只说明“计算发生在 GPU 上”，没有回答“到底省了多少”。要回答这个问题，比较必须满足三个条件：**同一个框架、同一份输入、同一个计时边界**。

这里用同一套 PyTorch 代码做一组对照：一组是开篇那种往返写法，另一组是权重留在 GPU 上的张量筛选。

将下面内容保存为 `torch_prune_roundtrip_demo.py`：

```python
"""给权重做剪枝，CUDA 到底加速了哪一步？（第8章 · TensorRT 部署）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch、NumPy。
"""
import time

import numpy as np
import torch
import torch.nn as nn


class DemoNet(nn.Module):
    """演示网络：256 → 128 → 64 → 10。"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)


def build_model(seed=0):
    """每次都用同一个种子构建，保证各次采样的输入完全一致。"""
    torch.manual_seed(seed)
    return DemoNet().cuda()


def prune_via_cpu(model, threshold):
    """开篇那种写法：权重在 GPU，筛选在 CPU 上完成。"""
    for name, param in model.named_parameters():
        if not name.endswith("weight"):
            continue
        weights = param.data.cpu().numpy()      # 设备 → 主机：同步拷贝
        mask = np.abs(weights) > threshold
        weights[~mask] = 0
        # 开篇写的是 .cuda()，隐含默认设备；这里按权重所在设备回传。
        param.data = torch.from_numpy(weights).to(param.device)


def prune_on_gpu(model, threshold):
    """权重留在 GPU，用张量运算完成筛选。"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not name.endswith("weight"):
                continue
            param.masked_fill_(param.abs() <= threshold, 0)


def measure(prune_fn, threshold, warmup=10, samples=50):
    """每次采样都新建模型，保证两条路径处理的是同一份未修改的权重。"""
    for _ in range(warmup):
        prune_fn(build_model(), threshold)
        torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        model = build_model()
        torch.cuda.synchronize()                 # 停表前先确保上一次工作已完成
        start = time.perf_counter()
        prune_fn(model, threshold)
        torch.cuda.synchronize()                 # 等 GPU 真正做完再停表
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95, method="linear"))


def main():
    threshold = 0.125

    # 先比正确性：两条路径必须给出同样的权重。
    model_cpu = build_model()
    model_gpu = build_model()
    prune_via_cpu(model_cpu, threshold)
    prune_on_gpu(model_gpu, threshold)
    for (name_a, pa), (name_b, pb) in zip(model_cpu.named_parameters(),
                                          model_gpu.named_parameters()):
        assert name_a == name_b
        # 演示权重由 PyTorch 随机初始化，不含 NaN，两条路径判定结果一致。
        torch.testing.assert_close(pa.detach(), pb.detach(), rtol=0, atol=0)
    print("两条路径结果一致。")

    p50_cpu, p95_cpu = measure(prune_via_cpu, threshold)
    p50_gpu, p95_gpu = measure(prune_on_gpu, threshold)
    print("原始往返写法（GPU→CPU→GPU）: p50=%.4f ms, p95=%.4f ms" % (p50_cpu, p95_cpu))
    print("GPU 张量筛选（权重留在 GPU）: p50=%.4f ms, p95=%.4f ms" % (p50_gpu, p95_gpu))
    print("p50 比值: %.2fx" % (p50_cpu / p50_gpu))


if __name__ == "__main__":
    main()
```

运行入口：

```bash
python torch_prune_roundtrip_demo.py
```

两条路径在自动求导上的处理不同（一边用 `.data` 绕过、一边用 `no_grad` 关闭），这不改变筛选结果，也不改变各自的数据往返量。真正不同只有一处：**权重要不要离开 GPU**。

- 原始写法：每个权重张量都要经历一次“设备 → 主机”的同步拷贝，再加一次“主机 → 设备”的回传；绝对值、比较、赋零都在 CPU 上完成。
- GPU 写法：读权重、算绝对值、比较、置零全在设备上完成，权重不出显存，也没有中间的主机数组。

计时边界对两条路径完全一致：从调用开始，到 `torch.cuda.synchronize()` 返回为止。这段时间包含 Python 调用开销、同步等待和全部数据搬运，正好对应“部署准备阶段真正付出的时间”。[1]

每次采样都重建模型，两条路径处理的输入因此完全一致。这是“从同一份未修改的原数组出发”在框架里的落地方式：置零只改变元素的值，不会把数组清空，重复采样也不会混入“已经剪过一遍”的输入。

两组数字打印出来后，可以这样读：

- 差距明显，说明在这条路径上往返搬运是主要成本；
- 差距不大，说明在这个规模下搬运不是主要矛盾，为这一步写自定义 kernel 就没有意义；
- 无论结果如何，它只描述**剪枝这一准备步骤**，与后续每次推理的延迟无关（后面一节展开）。

**验证状态**：脚本需要 PyTorch 和一块受支持的 NVIDIA GPU。本文只做了 Python 语法检查，没有在 NVIDIA GPU 上执行，因此不引用任何实测数字（未实测）。请在自己的机器上运行，用打印出来的 p50 与 p95 作为唯一依据。

**边界**：演示网络只有三个线性层，`endswith("weight")` 恰好只选中这三个权重张量，合计 41600 个元素。真实模型里这个条件还会命中归一化层、嵌入层等参数，量纲和稀疏性假设完全不同；换模型时应显式列出要处理的层，而不是沿用这个名字后缀。

## 05｜一个线程只处理一个权重，为什么适合并行？

对于阈值筛选，每个权重的处理只依赖它自己和同一个阈值。

处理第 7 个权重，不需要知道第 6 个是否被置零。于是可以让不同线程各自负责一个元素：读权重、算绝对值、比较、写结果。

连续权重可以按一维数组访问。CUDA 线程使用下面的索引找到自己的元素：[3]

```text
i = blockIdx.x × blockDim.x + threadIdx.x
```

其中，`blockIdx.x` 是线程块编号，`blockDim.x` 是每块线程数，`threadIdx.x` 是块内线程编号。

假设为了画图，每块只放 4 个线程，共有 10 个权重：

| 线程块 | 块内编号 | 得到的元素编号 | 处理范围 |
|---|---|---|---|
| 0 | 0、1、2、3 | 0、1、2、3 | 有效 |
| 1 | 0、1、2、3 | 4、5、6、7 | 有效 |
| 2 | 0、1、2、3 | 8、9、10、11 | 只处理 8、9 |

最后一个线程块没有装满，因此必须检查 `i < n`。否则线程 10、11 会访问数组之外的内存。

这里不需要共享内存，也不需要线程块内的 `__syncthreads()`：各线程读取各自元素、写入各自位置，没有互相等待的中间结果。GPU 内线程间不需要屏障，不代表 CPU 可以不等 GPU 完成就读结果。

下面的完整实验会使用每块 256 个线程。这是示例配置，不是对所有 GPU 的最优设置。

## 06｜独立实验：CPU、GPU 数组表达式与自定义 kernel

前面的 PyTorch 对照实验回答了“开篇那种往返写法要付出多少代价”。这一节换一个问题：**这项操作本身在 GPU 上怎么做，又怎样公平地测它。**

本节继续用 PyCUDA，但**放在独立 Python 进程里运行**，不把它的 `autoinit` 上下文与前文 PyTorch/TensorRT 的设备指针直接混用。两套框架的数字也不放进同一张表比较：PyTorch 一侧比的是“同一框架内的两条写法”，PyCUDA 一侧比的是“同一个操作在不同数据位置、不同实现下的成本”，这是两个不同的问题。

PyCUDA 的 `SourceModule` 可以编译 CUDA 源码，`GPUArray` 则提供驻留 GPU 的数组操作。[4]

为了公平，三种实现都从同一份原始数组计算一份新输出，原输入不修改。这样重复测量时，处理的始终是相同数据，避免把原始权重与已经置零的权重混在同一组比较中。

这里要说清一件事：阈值筛选只把满足 `|w| ≤ t` 的元素置零，数组本身不会因此变成全零。用同一个阈值重复执行属于幂等操作，保留下来的大权重仍然存在。重复处理已经剪过的权重会不会更慢，取决于被置零位置的数量是否影响掩码赋值的开销——这一点需要实测，本文不下结论。

| 路径 | 输入在哪 | 核心操作 | 输出在哪 |
|---|---|---|---|
| CPU 参考 | CPU | NumPy 阈值筛选 | CPU |
| GPU 数组表达式 | GPU | 绝对值、减阈值、条件选择 | GPU |
| 自定义 kernel | GPU | 一次逐元素读、判断、写 | GPU |
| kernel 往返调用 | CPU | H2D、kernel、D2H | CPU |

其中 GPU 数组表达式使用 `if_positive`：条件大于零就保留原值，否则选零数组。令条件为 `abs(w)-t`，就对应前面定义的规则。[5]

**环境与验证状态**：需要 NumPy、PyCUDA、受支持的 NVIDIA GPU、驱动、CUDA Toolkit 的 `nvcc` 及兼容的主机编译器。本文没有安装并验证这个组合，下面脚本仅通过 Python 语法检查，CUDA 字符串未实际编译；没有生成 GPU 运行结果。前面的 PyTorch 对照实验与这个脚本分开运行。

将下面内容保存为 `threshold_prune_demo.py`：

```python
"""给权重做剪枝，CUDA 到底加速了哪一步？（第8章 · TensorRT 部署）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import time
import numpy as np
import pycuda
import pycuda.autoinit  # 本实验独立建立上下文，不混用其他框架的指针。
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


CUDA_SOURCE = r"""
#include <math.h>
__global__ void threshold_prune(
    const float *input, float *output, int n, float threshold)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (unsigned int)n) {
        float w = input[i];
        output[i] = fabsf(w) > threshold ? w : 0.0f;
    }
}
"""


def cpu_reference(source, threshold):
    return np.where(np.abs(source) > threshold, source, np.float32(0))


def measure_wall(call, gpu=False, warmup=10, samples=50):
    for _ in range(warmup):
        call()
        if gpu:
            cuda.Context.synchronize()
    times = []
    for _ in range(samples):
        if gpu:
            cuda.Context.synchronize()
        start = time.perf_counter()
        call()
        if gpu:
            cuda.Context.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return {"samples": samples, "p50_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95, method="linear"))}


def main():
    # 编译/加载位于采样之外；缓存可能影响这一步，不能称为冷编译测量。
    start = time.perf_counter()
    module = SourceModule(CUDA_SOURCE)
    kernel = module.get_function("threshold_prune")
    print("源码编译/加载墙钟秒数:", time.perf_counter() - start)
    print("PyCUDA:", pycuda.VERSION_TEXT)
    print("GPU:", cuda.Context.get_device().name())
    threshold = np.float32(0.125)
    rng = np.random.default_rng(42)

    # 小规模、线程块边界、示例网络规模、大数组分别观察。
    for n in (0, 1, 255, 256, 257, 41600, 1000003, 4194304):
        source = rng.standard_normal(n).astype(np.float32)
        edges = np.array([-1, -threshold, -0.0, 0.0, threshold, 1], np.float32)
        source[:min(n, len(edges))] = edges[:min(n, len(edges))]
        expected = cpu_reference(source, threshold)
        if n == 0:
            assert expected.size == 0
            print("n=0：按约定返回空数组，不分配 GPU 内存或启动 kernel。")
            continue

        # 设备数组均为本次独立 PyCUDA 上下文中的连续 FP32 数组。
        d_input = gpuarray.to_gpu(source)
        d_output = gpuarray.empty_like(d_input)
        d_high = gpuarray.empty_like(d_input)
        d_zero = gpuarray.zeros_like(d_input)
        host_output = np.empty_like(source)
        grid = ((n + 255) // 256, 1, 1)

        def launch():
            kernel(d_input.gpudata, d_output.gpudata, np.int32(n), threshold,
                   block=(256, 1, 1), grid=grid)

        def high_level():
            criterion = abs(d_input) - threshold
            gpuarray.if_positive(criterion, d_input, d_zero, out=d_high)

        def round_trip():
            # 同步复制，包含传输但排除缓冲区分配和编译。
            cuda.memcpy_htod(d_input.gpudata, source)
            launch()
            cuda.memcpy_dtoh(host_output, d_output.gpudata)

        # 先正确性、后性能。get() 只用于检查，不放进驻留操作计时。
        launch()
        high_level()
        cuda.Context.synchronize()
        np.testing.assert_array_equal(d_output.get(), expected)
        np.testing.assert_array_equal(d_high.get(), expected)
        round_trip()
        np.testing.assert_array_equal(host_output, expected)

        print("\nn:", n, "CPU/GPU结果在本组输入上相同")
        print("CPU参考（含NumPy临时数组）:",
              measure_wall(lambda: cpu_reference(source, threshold)))
        print("GPU数组表达式（驻留、含临时数组与同步）:",
              measure_wall(high_level, gpu=True))
        print("自定义kernel（驻留、含提交与同步）:",
              measure_wall(launch, gpu=True))
        print("自定义kernel往返（含H2D/D2H）:",
              measure_wall(round_trip, gpu=True))


if __name__ == "__main__":
    main()
```

运行入口：

```bash
python threshold_prune_demo.py
```

41600 这个规模对应的是示例网络的权重元素总数，但脚本把它当成**一段连续数组**处理一次。真实模型里是三个独立张量、逐层调用，既多出每次调用的固定开销，也无法体现张量之间的差异。这里只匹配元素总量，不能把单次处理 41600 个连续元素的耗时当成整个模型逐层剪枝的耗时。

本例仅覆盖所生成的有限 FP32 数值与零，阈值固定为精确表示的 0.125；没有声称验证 NaN、Inf、次正规数或不同编译浮点选项下的全部语义。

`int n` 也限定了元素计数范围；这里最大测试规模远小于有符号 32 位上限。若要处理更大数组，应重新设计索引和计数类型，不能把本例无条件扩成通用算子。

## 07｜这段 kernel 里，省掉了什么？

核心只有两行：先读一个权重，再根据绝对值和阈值写出结果。

相邻线程访问相邻元素，输入和输出都是连续数组。这给合并访存提供了合适条件；不意味着它就已经达到硬件峰值。[6]

对照 GPU 数组表达式，`abs`、减法和条件选择需要产生并传递中间结果。自定义 kernel 将这段逻辑写在同一处，有机会减少中间数组与独立调用。

但结果是否更快，仍然取决于实现、分配行为、数组规模与设备。不能因为源码更短，就跳过测量；也不能把本实验中的数组表达式当成所有框架都没有融合的证据。

**归因边界**：即使自定义 kernel 更快，本实验也只能说明这两种写法的整体成本不同，不能单独量化“内核融合”贡献了多少。差异可能同时来自三处——中间数组的分配与释放、中间结果的读写、多次操作各自的提交开销。要把它们分开，需要再补一个复用预分配临时缓冲区的版本；本文不做这一步，也不对其中任何一项给出量化结论。

### 为什么没有共享内存？

共享内存适合某些线程间数据复用。这里每个权重只被对应线程读取并处理，没有展示出值得先搬进共享内存的复用关系。

加入共享内存会增加步骤，却未必减少全局访问。让优化措施服务于真实的数据依赖，比把熟悉的 CUDA 技术都加进来更有用。

## 08｜四行耗时，不能混成一个“加速比”

本例沿用第四篇的方法，先预热，再采样，用 CPU 墙钟配合完成等待。

GPU 两种驻留测试测的是**主机提交到 GPU 完成的同步调用耗时**，包含 Python 调用和同步，不是纯 kernel 执行时间。数组表达式的临时数组成本也在范围内。

CPU 路径包含 NumPy 的临时数组与新输出；GPU 路径提前准备了主要输入输出。因此，它们首先回答不同数据位置下的实现成本，不能据此宣称完成了只改变处理器、其他都完全不变的比较。

想判断“权重原本在 CPU，这次是否值得搬上 GPU”，应重点看含传输的往返路径，再与对应 CPU 路径比较。想判断“权重已经在 GPU，是否值得回 CPU”，应考察实际往返带来的额外成本。[6] 前面那组 PyTorch 对照实验正是第二种问法的答案：它比较的就是“权重已经在 GPU 又被送回 CPU 再传回来”和“权重留在 GPU”两种写法。

如果还要测 GPU 事件区间，可采用第四篇的 Event 方法；PyCUDA 的 `time_till` 返回毫秒，需要先保证结束事件完成。[7] 同样要注明是否包含传输、主机提交空隙，以及重复提交得到的是平均区间还是单次样本。

预热 10 次、采样 50 次只是脚本的教学设置，尤其 P95 仅是本次小样本估计。正式结论需要增加采样、重复不同顺序、记录 CPU/GPU 与驱动/Toolkit/PyCUDA/NumPy 版本，并观察温度和其他负载。

本文没有运行 GPU 实验，所以不会在表格里填“快多少倍”。即使实测结果表明小数组不值得上 GPU，也应保留。

## 09｜处理了权重，还没有证明模型更快

现在回到 TensorRT 部署链路。

本篇加速的候选对象是：**按照规则筛选并写出权重的操作**。如果它只在部署准备阶段运行一次，节省的是这一步的准备时间，不会自动从每次推理中再省同样的时间。

而且，本例输出数组与输入具有相同形状，仍是稠密存储。零值比例高，并不意味着后续稠密矩阵计算会自动跳过这些零。

示例中的三层网络 `256→128→64→10`，仅 weight 元素数为：

```text
256×128 + 128×64 + 64×10 = 41600
```

这也是脚本加入 41600 规模测试的原因；它与前几篇的两层演示网络不是同一个结构。无论其中多少数变成零，数组的这些位置仍然存在。零值比例不能直接代替参数数量下降、存储压缩率或推理加速比。

还要注意：41600 这个数字只对上权重元素总数，并没有模拟多层张量的组织方式。真实模型是三个独立张量、逐层调用，既多出每次调用的固定开销，也无法体现张量之间的差异。**一次处理 41600 个连续元素的耗时，不能直接视为整个模型逐层剪枝的耗时。**

PyTorch 剪枝教程通过权重与 mask 管理稀疏化；即使调用 `prune.remove` 固化剪枝结果，也不等于自动裁掉神经元并缩小网络维度。[2]

另外，已经生成的 TensorRT 引擎不会因你后来修改了 PyTorch 模型，就自动更新。本文流程应从修改后的模型重新导出和构建；专门的引擎 refit 机制不在本篇范围内。

至于如何让后续计算真正少做事，需要继续讨论结构裁剪、合适的稀疏存储或匹配的稀疏实现。例如 TensorRT 对 2:4 结构化稀疏有明确条件：它检查的是**指定方向上每组四个权重中至少有两个为零**，而不是“整个矩阵的零值达到一半”；即使满足这一条件，TensorRT 仍可能出于性能原因选择稠密实现。任意阈值产生的零分布，与这种分组约束并不是一回事。[8]

这些内容留给下一篇，避免把“筛选很快”直接写成“模型已加速”。

## 10｜什么时候值得自己写这段 CUDA？

如果只是偶尔处理一次小模型，先使用可读的框架张量操作通常更容易维护。若大量权重已经驻留 GPU、筛选又确实是需要优化的步骤，可以进一步评估自定义 kernel，重点看它是否减少了中间处理及不必要的数据往返。

前面那组 PyTorch 对照实验正好给出一个判断依据：如果“绕回 CPU 再传回来”和“留在 GPU”之间的差距明显，往返搬运就是主要成本，值得优化；如果差距很小，这一步就不必专门写 kernel。

正确的推进顺序是：先定义阈值语义，再验证边界和数值一致；先明确数据原本在哪里，再选择性能比较的范围；最后回到部署链路，判断这项局部收益是否值得保留。

这一篇能交付的结论，是“GPU 如何完成这项筛选，以及怎样验证它”。模型质量是否保持、TensorRT 推理是否更快，仍然需要各自的证据。

下一篇，我们就接着问：权重已经有一半变成零，模型为什么可能一点也没快？

## 参考资料

1. [PyTorch Tensor.masked_fill_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.masked_fill_.html)；[CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)：掩码填值与 CUDA 张量执行。
2. [PyTorch Pruning Tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)；[prune.remove](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.prune.remove.html)：mask、重参数化与固化剪枝结果。
3. [NVIDIA CUDA: Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)；[Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)：线程索引、层次与同步。
4. [PyCUDA Tutorial](https://documen.tician.de/pycuda/tutorial.html)：SourceModule 编译、启动与 GPUArray 使用。
5. [PyCUDA GPU Arrays](https://documen.tician.de/pycuda/array.html)：绝对值、条件选择和设备数组接口。
6. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：热点识别、数据传输成本、访存与性能评价。
7. [PyCUDA Device Interface](https://documen.tician.de/pycuda/driver.html)：同步复制、事件和毫秒计时。
8. [TensorRT 11.4 Data Formats and Tensors](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/inference-library/data-formats-tensors.html)：结构化稀疏条件和执行实现选择。
