# 一组线程读同一个数，GPU 有什么捷径？

> **GPU 内存优化 · 第 4 篇**  
> 数据只读，还不足以决定该放在哪里。对常量内存而言，同一组线程是否读取相同地址，往往更关键。

> **工程深度｜验证状态：** L2 工程｜概念验证（访存视角）
假设你正在给一段传感器数据做平滑处理：每个采样点，都要与前面几个点按固定权重组合。数据有很多，权重只有五个，而且所有采样点使用同一套权重。

把工作交给 GPU 后，一个线程负责一个输出。大家处理的是不同位置，却会反复遇到相同的问题：“这一步的系数是多少？”

上一篇，我们用共享内存，让线程共同取数、在块内复用。面对这五个只读系数，是否还有更适合的方式？

CUDA 的**常量内存（constant memory）**值得考虑。但它的优势有明确条件：**同一个 warp 中的线程，在同一条读取指令上，最好只需要少数几个地址。**这里的 warp 是 NVIDIA GPU 中由 32 个线程组成的执行分组，而不是整个 GPU 的所有线程。[1]

本篇用固定系数滤波解释这种访问模式，再用一个小查表实验观察地址分散后的变化。场景、公式与访问表均用于教学；代码未完成 GPU 实测，不提供虚构的耗时和加速比。资料核对日期：2026 年 10 月 20 日。

## 01｜常量内存的“常量”，是什么意思

先避免一个误会：常量内存不是一整块比共享内存更大的片上高速空间。

CUDA 文档说明，常量内存的后备存储位于设备内存，通过常量缓存服务读取；命中缓存和未命中缓存的访问成本不同。[1]

对 GPU 内核而言，它是只读的。主机端可以用适当接口准备或更新数据，但不能在仍有内核并发使用相关常量时随意改写。[1]

因此，“常量”不代表程序启动后永远不能改变，而是需要区分**哪一端可以更新、采用什么接口、何时更新**。

它也不等于“把普通数组标成只读”。例如，不修改某个全局内存数组，并不会自动让它变成常量内存数组。

可以先用一张表对照上一篇：

| 问题 | 共享内存：本文系列采用的块内用法 | 常量内存 |
|---|---|---|
| 谁组织数据 | 线程协作加载和使用 | 主机准备，内核读取 |
| 内核能否写入 | 可以 | 不可以 |
| 适合观察什么 | 数据复用、访问重排 | 少量只读参数的一致读取 |
| 需要避免什么 | 协作失序、bank conflict、资源过量 | 地址过度分散、容量或更新方式不合适 |

表格只用于区分机制，不构成固定的速度排名。

## 02｜真正的线索：不是大家都读，而是读同一个地址

假设一个 warp 的 32 个线程，都在处理各自的采样点。滤波循环走到第 2 个系数时：

| 线程 | 读取的输入 | 读取的系数 |
|---|---|---|
| 0 | `x[i0 - 2]` | `weights[2]` |
| 1 | `x[i1 - 2]` | `weights[2]` |
| 2 | `x[i2 - 2]` | `weights[2]` |
| … | 各自位置的数据 | `weights[2]` |
| 31 | `x[i31 - 2]` | `weights[2]` |

输入地址不同，系数地址却相同。常量缓存路径能够利用这种**广播**：同一组线程请求同一个位置，不必将其视为 32 个不同地址分别处理。[2]

现在换一种任务，让每个线程查自己的系数：

| warp 内的系数索引 | 不同地址数量 | 需要关注什么 |
|---|---:|---|
| 0、0、0、0、… | 1 | 适合利用广播 |
| 0、1、2、3、0、1、2、3、… | 4 | 同一次请求包含多个地址 |
| 0、1、2、3、…、31 | 32 | 地址高度分散 |

NVIDIA 文档指出，warp 的常量内存请求会按不同地址拆分，地址越多，可能需要处理的请求越多。[1][2]

这与第二篇的全局内存合并访问有所不同：对全局内存，32 个相邻 FP32 地址通常是值得追求的访问模式；对常量内存，同一条指令中的 32 个不同地址，不能获得“大家读同一处”的优势。

这里说的是**一次读取指令中参与线程的地址关系**。不能因为整个程序最终都会读完同一张表，就认定它适合广播；也不能把 32 个地址直接换算成整个程序慢 32 倍。

还有一个容易跳过的环节：表很小时，读者可能会问——数据都命中常量缓存了，不同地址还有什么影响？答案是：**即使数据都命中常量缓存，同一请求里的不同地址仍需分别服务；小表有利于缓存命中，却不能消除地址分散带来的处理代价。**[2] 这也是官方最佳实践把"缓存命中"与"不同地址访问"分开讨论的原因。

## 03｜让五个系数真正参与一次计算

给教学场景一个明确的定义。输入序列为 x，输出为 y：

\[
y_i=\sum_{k=0}^{4}w_kx_{i-k}
\]

当 \(i-k<0\) 时，将该输入视为零。系数选为：

\[
w=[0.40,\ 0.25,\ 0.20,\ 0.10,\ 0.05]
\]

这是一个简单的**因果有限冲激响应滤波器（FIR）**：每个输出只依赖当前与过去有限个输入。这里用它说明内存访问，不把这组系数当作任何具体传感器的最佳方案。

这类"小系数表被海量线程反复读"的形态在工程里很常见：音频均衡器的滤波系数、传感器去噪器的核函数，都属于这一类。神经网络里也有看似相近的例子——LayerNorm 的缩放参数、Linear 层的偏置同样只读、会被反复取用，但若一个 warp 的线程分别读取不同特征对应的参数，就不具备这里的同地址广播条件。**只读和复用，并不能单独决定存储方式；同一条指令中相邻线程读的是同一地址还是不同地址，才是本篇的判断标准。**

若输入开头是 `[1, 3, 2, ...]`，按公式计算：

\[
y_0=0.4,\qquad
y_1=0.4\times3+0.25\times1=1.45
\]

\[
y_2=0.4\times2+0.25\times3+0.2\times1=1.75
\]

这些是公式推导值，实际 FP32 输出可能有微小舍入差异。

一个线程计算一个 y。循环变量 k 在本轮相同，所以对于参与该次读取的线程，系数索引一致。序列开头因补零条件，参与计算的线程可能较少，但参加同一次系数读取的线程仍使用同一索引。

本篇只调整系数的存储与读取方式，不加入共享内存优化输入窗口，也不把多项优化叠在一起解释效果。

## 04｜Numba-CUDA 中，应该怎样准备常量

一个常见的错误写法是：让第一个 GPU 线程写入常量数组，再调用 `syncthreads()` 等其他线程使用。这既不符合内核只读语义，也不能靠块内屏障完成跨线程块初始化——常量数据不能由设备端这样修改，块内屏障也不能同步整个网格。[1][3]

Numba-CUDA 支持的一个方式是：在主机端准备 NumPy 数组，在内核中使用 `cuda.const.array_like()`。官方 API 说明，它在**编译时**将数组内容复制到常量内存。[3]

下面的第一段代码定义两种相同计算：

```python
"""一组线程读同一个数，GPU 有什么捷径？（第7章 · 访存与传输）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：Numba、NumPy。
"""
import numpy as np
from numba import cuda, float32

THREADS = 256
TAPS = 5
WEIGHTS = np.array([0.40, 0.25, 0.20, 0.10, 0.05], dtype=np.float32)


@cuda.jit
def filter_global(x, weights, y):
    i = cuda.grid(1)
    if i < x.size:
        acc = float32(0.0)
        for k in range(TAPS):
            pos = i - k
            if pos >= 0:
                acc += weights[k] * x[pos]
        y[i] = acc


@cuda.jit
def filter_constant(x, y):
    weights = cuda.const.array_like(WEIGHTS)
    i = cuda.grid(1)
    if i < x.size:
        acc = float32(0.0)
        for k in range(TAPS):
            pos = i - k
            if pos >= 0:
                acc += weights[k] * x[pos]
        y[i] = acc
```

第一种把普通设备数组作为参数传入；第二种使用编译时嵌入的常量数组。两种都计算相同公式，既没有在内核中修改系数，也不需要让线程们先协作加载系数。

**不要在编译后改一改 `WEIGHTS`，就假设已经编译的内核会自动读到新值。**这里不是运行时传参接口。需要频繁切换系数时，普通设备数组可能更方便；如果专门使用 CUDA C++ 的常量符号，也可通过 `cudaMemcpyToSymbol()` 等接口更新，但必须管理好更新与使用的先后关系。[4]

还有一个实验边界：五个系数很少，循环范围也固定。编译器可能展开循环、折叠常量或改变最终指令。因此，即使两个版本耗时不同，也不能仅凭这段源码，将差异全部归因于常量缓存广播。

这不是实验失效，而是提醒我们：**源代码表达了意图，最终指令和运行状态才决定具体执行。**

## 05｜再加一个对照：地址分散以后会怎样

两个实验的任务分工先说清楚：**滤波实验**展示如何把固定系数接入真实计算，并比较两种实现；**查表实验**观察地址模式变化时两种实现的表现，不测量纯粹的常量缓存延迟。

为了更直接地观察读取模式，再准备一个包含 256 个 FP32 元素的小表。它占 1024 字节。每个线程从中取一个系数，与自己的输入相乘：

\[
y_i=x_i\times\mathrm{table}[\mathrm{index}(i)]
\]

用参数 u 控制一个完整 warp 内的不同地址数：

```python
# 37 只是让不同 warp 改变查表起点，不是硬件参数。
index = ((i // 32) * 37 + (i % 32) % u) % 256
```

这里：

- `i // 32` 是分组编号，让不同 warp 不总是读同一段表。
- `i % 32` 是组内位置。
- 再 `% u`，让组内索引重复。
- 最后 `% 256`，保证地址在表内。

本文使用 256 线程的一维块，块大小是 32 的整数倍，因此这样的分组与 warp 对齐。对于完整 warp，u=1、4、32 分别产生 1、4、32 个不同地址；最后一个不完整 warp 可能更少。每个完整 warp 内起点相同，组内偏移才决定不同地址数。三种设置在源码层面分别构造 1、4、32 个不同地址；若最终对应同一次 warp 常量读取请求，按官方描述会产生相应的地址拆分。[1][2]

| 设置 | 源码中一个完整 warp 的不同索引数 | 对应的地址关系 |
|---|---:|---|
| u=1 | 1 | 全部线程读同一地址 |
| u=4 | 4 | 同一请求包含四个地址 |
| u=32 | 32 | 每个线程各不相同 |

这是并列对照，不是执行顺序；三种设置之间没有先后关系。

每个 u 都分别测试普通设备数组和常量数组。**先比较同一个 u 下的两个版本**，再观察 u 变化后的趋势。不同 u 会改变取到的系数，不能把它们当作同一计算结果的优化前后。

这个微型任务还有一个限制：每个输出只查一次表，完整耗时中还有输入读取、输出写入、索引计算与启动开销。它能帮助观察现象，却不是隔离所有其他因素后的“常量缓存延迟测试”。如果测不出明显差异，不能据此否定广播机制。

## 06｜怎样测，才不会把想象当成绩

文末给出补齐脚本。它先按 CPU 参照检查结果，再测量已预热的 GPU 工作区间。

**环境与验证状态：**需要 NumPy、NVIDIA 维护的 Numba-CUDA，以及兼容的驱动和 CUDA 依赖；安装方式以官方文档为准。[5] 本文生成环境没有 Numba-CUDA，未完成 CUDA 编译、GPU 执行或性能测量。已检查 Python 语法，并在 CPU 上枚举线程索引，验证滤波边界、查表范围与完整 warp 的不同地址数；这些检查不证明常量内存的实际生成或广播性能。

测量采用下面的条件：

| 项目 | 设置 |
|---|---|
| 基线 | 完成同一公式、使用普通设备系数数组的版本 |
| 数据类型 | 输入、系数与输出均为 FP32 |
| 数据位置 | 计时前已分配并复制到 GPU |
| 正确性 | 滤波与 CPU float64 参照做容差比较；查表与对应索引结果比较 |
| 预热与采样 | 预热 3 次；采 5 组，每组 20 次 |
| 计时 | 默认流上的 CUDA Event，报告组均值中位数与范围 |
| 不包含 | 数据准备、编译、主机与设备复制 |

Numba 的事件计时返回毫秒，需要等待结束事件完成再读取。[6] 很短的任务可能受事件分辨率、主机提交空隙影响；重复访问同一小表也会形成较热的缓存。本实验不代表冷启动或端到端应用表现。

建议这样记录自己的结果：

| 任务与规模 | warp 内不同地址数 | 普通数组 / ms | 常量数组 / ms | 正确性 |
|---|---:|---|---|---|
| 固定系数滤波 | 每次系数读取为 1 | 待测 | 待测 | 待验证 |
| 查表，规模自填 | 1 | 待测 | 待测 | 待验证 |
| 查表，同上 | 4 | 待测 | 待测 | 待验证 |
| 查表，同上 | 32 | 待测 | 待测 | 待验证 |

**进一步确认读取路径（进阶，可跳过）：**若要把耗时差异进一步归因，可以在编译后使用内核的 `inspect_asm()` 查看 PTX，或在环境支持时查看 SASS；Numba-CUDA 提供这些检查接口，但依赖相应工具环境。[3] 需要确认目标数组的实际读取，而不是只看到内核参数相关的常量空间就下结论。PTX 仍是中间表示，必要时结合最终指令与 Nsight Compute 的源码/指令关联分析。[7] 只想理解机制的读者，记住"不能仅凭源码声明判断实际执行路径"即可。

普通全局读取也有缓存，编译器还可能利用地址一致性。最终看到“两种都快”或“差异不大”，都可能是合理结果。若要判断是否改善了实际应用，最后还要回到包含其他步骤的完整流程。

## 07｜哪些数据值得放进去

CUDA 文档列出的常量内存空间容量是 64 KiB；常量缓存的工作集容量又是另一件事，不能将两者等同，更不能推断“放得进去就全程命中”。[1][2]

五个 FP32 系数只有 20 字节，空间很小。但“神经网络的权重训练完就只读”，并不足以把整份模型放进去：模型参数可能远超容量，而且同一 warp 未必在同一时刻读取相同权重地址。

判断时，依次问四个问题：

1. **小不小？**数据规模是否适合这个受限空间？
2. **读法对不对？**同一条指令中，warp 内有多少不同地址？
3. **怎么更新？**固定嵌入、按需更新还是每批切换，哪种接口更合适？
4. **是否值得？**这个读取是不是瓶颈，收益能否覆盖实现与维护成本？

常量内存适合成为一个有条件的选项，而不是所有只读数组的默认归宿。固定的小型滤波系数，提供了一个容易理解的候选；每个线程各查各的表，则需要更谨慎地验证。

前三篇分别看了空间占用、全局访问与块内协作，本篇再补上一种一致读取的路径。接下来，如果 GPU 本身取数已经合理，却仍在等 CPU 送数据，就该沿着主机到设备的传输链路继续查了。

## 附：补齐查表与运行入口

将下列代码接在前面的滤波代码之后，保存为 `constant_memory_lab.py`，执行 `python constant_memory_lab.py`。系数在首次编译前准备，整个实验期间不修改。脚本会打印主要版本与设备信息，请另行记录驱动版本。

```python
from importlib.metadata import version
from statistics import median

TABLE_SIZE = 256
TABLE = np.random.default_rng(7).uniform(0.1, 1.0, TABLE_SIZE).astype(np.float32)


@cuda.jit
def lookup_global(x, table, y, u):
    i = cuda.grid(1)
    if i < x.size:
        # 37 只是让不同 warp 改变查表起点，不是硬件参数。
        index = ((i // 32) * 37 + (i % 32) % u) % TABLE_SIZE
        y[i] = x[i] * table[index]


@cuda.jit
def lookup_constant(x, y, u):
    table = cuda.const.array_like(TABLE)
    i = cuda.grid(1)
    if i < x.size:
        index = ((i // 32) * 37 + (i % 32) % u) % TABLE_SIZE
        y[i] = x[i] * table[index]


def measure(kernel, blocks, args):
    for _ in range(3):
        kernel[blocks, THREADS](*args)
    cuda.synchronize()
    start, end = cuda.event(timing=True), cuda.event(timing=True)
    times = []
    for _ in range(5):
        start.record()
        for _ in range(20):
            kernel[blocks, THREADS](*args)
        end.record()
        end.synchronize()
        times.append(cuda.event_elapsed_time(start, end) / 20)
    return median(times), min(times), max(times)


def check_pair(label, pair, blocks, expected, out, benchmark):
    for name, kernel, args in pair:
        kernel[blocks, THREADS](*args)
        cuda.synchronize()
        got = out.copy_to_host()
        # 容差针对本脚本的 FP32 输入；不是任意算法的误差界。
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
        print(f"  {label}/{name}: correctness PASS")
    if benchmark:
        for name, kernel, args in pair:
            mid, low, high = measure(kernel, blocks, args)
            print(
                f"  {label}/{name}: median={mid:.6f} ms"
                f", group means range=[{low:.6f}, {high:.6f}] ms"
            )


def run_case(n, benchmark=False):
    host = np.random.default_rng(42).random(n, dtype=np.float32)
    x = cuda.to_device(host)
    y = cuda.device_array(n, dtype=np.float32)
    dw, dt = cuda.to_device(WEIGHTS), cuda.to_device(TABLE)
    blocks = (n + THREADS - 1) // THREADS
    print(f"N={n}")

    expected = np.convolve(
        host.astype(np.float64), WEIGHTS.astype(np.float64), mode="full"
    )[:n]
    pair = [
        ("global", filter_global, (x, dw, y)),
        ("constant", filter_constant, (x, y)),
    ]
    check_pair("filter", pair, blocks, expected, y, benchmark)

    positions = np.arange(n, dtype=np.int64)
    for u in (1, 4, 32):
        indices = ((positions // 32) * 37 + (positions % 32) % u) % TABLE_SIZE
        expected = host * TABLE[indices]
        pair = [
            ("global", lookup_global, (x, dt, y, u)),
            ("constant", lookup_constant, (x, y, u)),
        ]
        check_pair(f"lookup/u={u}", pair, blocks, expected, y, benchmark)


def main():
    if not cuda.is_available():
        raise SystemExit("需要可用的 NVIDIA CUDA 环境。")
    print("NumPy:", np.__version__)
    print("Numba:", version("numba"))
    print("Numba-CUDA:", version("numba-cuda"))
    gpu = cuda.get_current_device()
    print("GPU:", gpu.name, "compute capability:", gpu.compute_capability)
    print("FP32, block=256, warmup=3, repeats=20, samples=5")
    for n in (17, 257):
        run_case(n)
    for n in (1 << 18, 1 << 20):
        run_case(n, benchmark=True)


if __name__ == "__main__":
    main()
```

默认先测普通数组，再测常量数组。差异很小时，应交换顺序并重复采样，检查缓存、GPU 状态和其他任务影响。脚本不打印“保证有效”的加速结论，也不将 CPU 参照的耗时与 GPU 比较。

## 参考资料

下列资料均来自 NVIDIA 或 NVIDIA 维护的 Numba-CUDA。CUDA 固定版本链接用于明确核验口径，不表示它是当前最新版本。

1. [NVIDIA：CUDA C++ Programming Guide 12.8 — Constant Memory、变量限定符与技术规格](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html) — 存储位置、缓存、线程分组、访问拆分、只读及更新约束。
2. [NVIDIA：CUDA C++ Best Practices Guide 12.8 — Constant Memory](https://docs.nvidia.com/cuda/archive/12.8.2/cuda-c-best-practices-guide/index.html#constant-memory) — 小量不同地址、广播与容量。
3. [NVIDIA Numba-CUDA：CUDA Kernel API](https://nvidia.github.io/numba-cuda/reference/kernel.html) — `cuda.const.array_like()`、同步与编译产物检查接口。
4. [NVIDIA：CUDA Runtime API — cudaMemcpyToSymbol](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__MEMORY.html) — 主机端向常量或全局符号复制数据的接口。
5. [NVIDIA Numba-CUDA：Installation](https://nvidia.github.io/numba-cuda/user/installation.html) — 安装与环境依赖。
6. [NVIDIA Numba-CUDA：CUDA Host API — Events](https://nvidia.github.io/numba-cuda/reference/host.html#events) — 事件记录、等待与毫秒计时。
7. [NVIDIA Nsight Compute：工具文档](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) — 性能分析及源码、指令关联查看。

材料基础：用户提供的《第 8 章 GPU 内存优化》图片文字转写稿。本篇重写常量内存示例，用固定系数滤波与地址模式对照说明适用边界，未沿用原稿中未经核验的运行时间和误差输出。
