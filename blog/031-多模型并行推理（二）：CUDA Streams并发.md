# 开了 4 条 CUDA 流，GPU 就能同时推理 4 批图片吗？

> **多模型与多请求并发推理系列 · 第 2 篇｜CUDA Streams 与计算并发**  
> **一句话：CUDA Stream 不是把一张 GPU 切成几张小 GPU，而是告诉设备哪些任务可以不必相互等待。能不能真正同时跑，要看硬件资源和任务依赖。**
>
> 阅读路线：先认识 Stream → 看懂 GPU 时间轴 → 写出正确的 PyTorch 多流代码 → 用实验判断有没有收益 → 学会在 Nsight Systems 中验证。

<!-- more -->

> **本篇边界：** 本篇研究**同一个模型的多批次并发**，用来解释多模型服务同样会遇到的执行机制；它**不直接证明**多个不同模型共享 GPU 的性能收益。

> **工程深度｜验证状态：** L2 工程｜实验验证（建议上机跑通）
## 01｜四条车道，为何未必比一条车道快？

上一回，我们把智慧园区的监控系统拆成了人员检测、车辆识别、异常事件检测三个 AI 模型。我们发现：把独立模型同时提交，并不意味着它们一定能在一张 GPU 上同时计算。

现在把镜头拉近：先不管三个不同模型，而是只看园区里的一个**小型 CNN**。20 路摄像头不断上传图像，服务端把画面裁成若干图像块（Patch），交给这个 CNN 判断类别，再把结果组成批次（Batch）送去做前向计算。

本文使用的是一张随机初始化的十分类 CNN（输出 `[B,10]` logits），专门用来观察不同批次的 GPU 执行方式。真实的人员检测模型算子更复杂，但同样要处理流间依赖与资源竞争——本篇关心的正是这部分。

负责优化的工程师发现，代码里的推理循环是这样的：

```python
for batch in batches:
    output = model(batch)
```

“这里是不是只能一批接一批？”他创建了四个 CUDA Streams，把不同批次轮流交给它们。程序看起来可以一次提交更多计算。（顺带说明：更早的记录里曾出现过一次接近两倍的加速，但那组数字同时改变了流和拷贝行为，**不能归因于多流**，第 06 节会专门拆解。）

问题来了：**GPU 真的同时跑了四个 CNN 吗？还是 CPU 只是更快地把四份工作放进了队列？**

这正是 CUDA Streams 最容易被误解的地方。

### 三个容易混为一谈的“同时”

- **同时提交**：CPU 在前一个 GPU 任务尚未完成时，继续提交其他任务。
- **有资格同时执行**：不同任务之间没有依赖，且位于允许并发的执行流中。
- **真的同时计算**：GPU 在同一段时间里执行了来自不同流的 Kernel。

只有最后一个，才是我们真正要在时间线上验证的 **Kernel overlap**。

## 02｜CUDA Stream 是什么？先记住一句话

**一个 Stream 是一条按顺序执行 CUDA 命令的序列。不同 Stream 之间可以存在并发机会，但不承诺并发。**

例如，Stream 0 上依次提交三条命令：

```text
Stream 0： [Kernel A] → [Kernel B] → [Kernel C]
                必须遵守同一流内的先后顺序
```

再创建 Stream 1，把另一组独立工作交给它：

```text
时间向右 ─────────────────────────────────→

CPU 提交：  A0  A1  A2  B0  B1  B2  ...  (提交≠完成)

Stream 0： [A0────][A1────][A2────]
Stream 1：     [B0────][B1────][B2────]

GPU 实际：  是否如上重叠？必须测量，不能靠画图断定。
```

请注意：**Stream 不是 CPU 线程，也不是 GPU 的一个 SM，更不是一块独占的显存。**

可以把 Stream 想成餐厅的“订单轨道”：同一轨道上的订单保持先后顺序；不同轨道可把独立订单递给厨房。但厨房灶台只有那么多，订单轨道越多，并不代表灶台越多。

### 一个关键的时序对照

假设一张 GPU 的某个 CNN 批次计算耗时为 6 ms，四批都需要同样的工作量。这是纯教学示意，不代表真实设备实测。

```text
情形 A：完全串行（示意）
Stream 0 |[Batch 0: 6][Batch 1: 6][Batch 2: 6][Batch 3: 6]|
         0                                                      24 ms

情形 B：理想完全重叠（示意）
Stream 0 |[Batch 0: 6]|
Stream 1 |[Batch 1: 6]|
Stream 2 |[Batch 2: 6]|
Stream 3 |[Batch 3: 6]|
         0             6 ms
```

第二种图只是理想下界的直观表达，不是 Stream 的自动效果：在相同 GPU 上，四个任务争用计算单元、寄存器、共享内存、缓存和显存带宽。

如果每批已经单独吃满 GPU，四条流未必比一条更快；甚至可能因为排队和竞争而增加单批延迟。

官方依据：NVIDIA CUDA Programming Guide 的 *Asynchronous Execution* 与 *Stream Ordering* 章节强调流内有序、流间实际并发受可用资源制约。见文末参考资料 [1][2]。

## 03｜GPU 到底在争什么？

CNN 前向传播并非“一次 GPU 操作”，而是多个算子相继执行：卷积、激活、池化、全连接。框架可能通过 cuDNN 等库启动一个或多个 CUDA Kernel。

```mermaid
flowchart LR
    X[一批 32×32 图像] --> C1[Conv 3→16]
    C1 --> R1[ReLU / Pool]
    R1 --> C2[Conv 16→32]
    C2 --> R2[ReLU / Pool]
    R2 --> FC[Flatten / FC]
    FC --> Y[10 类 logits]
```

CUDA Stream 调度的是 GPU 命令的顺序，并不直接把某个 CNN 固定在一个专属计算区。是否能交叠，受至少四类资源影响：

| 资源或约束 | 为什么重要 | 可能发生什么 |
| --- | --- | --- |
| SM / 计算执行资源 | Kernel 的线程块需要在 SM 上驻留和执行 | 单个 Kernel 占用太充分，其他 Kernel 难以插入 |
| 寄存器与 shared memory | 每个线程块有资源配额 | 即便有空闲计算能力，也可能因为片上资源不足不能同时驻留 |
| 显存带宽与缓存 | 多个 CNN 同时读写激活和权重 | 并行任务相互争带宽，速度不增反降 |
| 数据与同步依赖 | 下一步必须等前一步输出 | 即便放在不同流里，也不能违反正确性要求 |

**“开多流”解决的是表达并发机会的问题；“获得加速”是硬件执行与应用负载共同决定的结果。**

### 为什么小任务更可能受益？

如果一个批次很小，CNN 启动的部分 Kernel 可能没有充分利用 GPU。其他 Stream 的独立工作可能填补空隙。但这并非普适规律：太小的任务也可能受到 CPU 提交开销、Kernel 启动开销影响，导致增加流数没有意义。

相反，如果一个大批次已经充分利用 GPU，再增加并发，不会凭空产生算力。

所以实验不应该只问“4 条流比 1 条快多少”，而应该问：**在什么 Batch Size、模型规模和 GPU 上，增加流数才有收益？**

## 04｜先避开一个陷阱：我们本篇只测计算，不测数据搬运

前面那份 CIFAR-10 示例使用了：

```python
with torch.cuda.stream(stream):
    x_device = x.to('cuda', non_blocking=True)
    out = model(x_device)
    results.append(out.cpu())
```

它把三种事情混在一起：CPU→GPU 传输、GPU 上 CNN 前向、GPU→CPU 返回结果。尤其 `out.cpu()` 的默认阻塞式复制可能迫使主机等待前面操作完成，削弱本来希望实现的跨批次重叠。

另外，给 `DataLoader` 设置 `pin_memory=True`，并不会让另外调用 `torch.randn(...)` 创建的普通 CPU 张量自动变成页锁定内存。

为避免“到底是谁带来的加速”说不清，本篇做一个干净的对照：

```text
准备阶段（不计时）：
    生成输入 → 放入 GPU 显存 → 加载模型 → 按配置创建 Streams → 预热

测试阶段（计时）：
    同一 CNN + 同一组 GPU 输入 + 复用同一组 Streams
       ├─ 对照：1 条 Stream
       └─ 实验：2 / 4 / 8 条 Stream
    全部完成后统一同步、读取结果
```

这意味着：**本篇研究 GPU 内部的计算并发，暂不评价 H2D/D2H 传输优化。** 数据传输与计算流水线将留给系列第 4 篇。

关于计时口径也说清楚：本实验比较的是**输入已在显存中的稳态推理完成时间**，计时区间包含 CPU 提交、必要的流间依赖、前向计算和最终同步，**排除模型加载、输入上传和 Stream 创建**。为此，Streams 在准备阶段按配置各创建一次，预热和正式测量复用同一组对象，而不是每轮重新创建。

## 05｜代码主角：一个足够简单的 CNN

仍然沿用最初示例的 SimpleCNN，输入是 `[B,3,32,32]`，输出是 `[B,10]` 的分类 logits。为了让任何有 CUDA GPU 的读者先复现实验，本篇不要求先下载 CIFAR-10，也不要求训练五个 Epoch。

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(), nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

模型权重虽然是随机初始化的，但**同一个模型、同一份输入**在两种调度方式下，输出应在**合理的数值容差内一致**（本实验用 `torch.allclose` 校验），而不是要求逐位相同。这足以验证计算一致性和测试执行机制；它不构成 CIFAR-10 分类准确率实验。如果要讨论识别精度，必须另外加载训练好的权重，并用真实测试集评估。

还有一句推理前提别漏掉：模型要先 `.eval()`。

```python
model = SimpleCNN().to(device).eval()
```

`torch.inference_mode()` 主要管理自动求导相关开销，**不会**自动把 Dropout、BatchNorm 切换成评估行为。本例没有这两层，所以数值不受影响，但写多流代码时应当养成 `eval()` 的习惯。

### 第一个版本：单流基线

```python
@torch.inference_mode()
def inference_single(model, batches):
    # batches 已经位于同一张 GPU 的显存中
    return [model(x) for x in batches]
```

这里没有 `.cpu()`，也不在每批后调用 `synchronize()`。CPU 提交工作可能是异步的，但同一 Stream 上的前向操作仍遵循提交顺序。

### 第二个版本：多流提交

下面是完整脚本中最关键的实现。它不是让一个模型变成四个不同模型，而是**让同一个 CNN 在多个 Stream 上处理彼此独立的输入批次**。

先分清两个最容易混的操作——它们解决的是完全不同的两个问题：

| 操作 | 它解决的问题 | 通俗理解 |
| --- | --- | --- |
| `wait_stream` | 使用数据前，生产工作是否完成 | 等食材准备好再开做 |
| `record_stream` | 异步使用尚未结束，这块内存会不会被回收复用 | 菜还没做完，别把案板收走 |

`streams` 参数在准备阶段由 `make_streams()` 按配置创建一次并复用（见第 04 节），所以创建成本不计入计时：

```python
def make_streams(num_streams, device):
    if num_streams <= 1:
        return []                         # 1 条流：用当前流即可
    return [torch.cuda.Stream(device=device)
            for _ in range(num_streams)]


@torch.inference_mode()
def run_batches(model, batches, streams, device):
    if not streams:                       # 1 条流：在当前流上顺序执行
        return [model(x) for x in batches]

    producer = torch.cuda.current_stream(device)
    for stream in streams:                # 消费者流等待输入准备好
        stream.wait_stream(producer)

    outputs = []
    for i, x in enumerate(batches):
        stream = streams[i % len(streams)]
        with torch.cuda.stream(stream):
            x.record_stream(stream)
            outputs.append(model(x))

    for stream in streams:                # 原流等待所有消费者
        producer.wait_stream(stream)

    return outputs
```

这段代码的价值主要体现在四个位置：

**① `streams[i % len(streams)]`：轮流派发，而不是轮流等待。** 以 4 条流为例，`i % 4` 的结果依次为 0、1、2、3、0、1……；它决定任务在哪条流排队，不代表哪一个 GPU 核心负责计算。

**② `with torch.cuda.stream(stream)`：改变后续 CUDA 操作的提交位置。** 上下文中的 GPU 运算排入指定流；Python `with` 自身不是“开启一个 GPU 线程”。

**③ `stream.wait_stream(producer)`：建立跨流先行依赖。** 这里 `producer` 取的就是**当前流**——也就是假设输入与模型权重已经在该流上准备好。消费者流必须在所需数据准备好后才使用它们。`wait_stream` 为之后排入该流的工作添加等待关系，而不是让 CPU 阻塞等待。保留显式依赖有助于讲明正确语义。

**④ `x.record_stream(stream)`：记录异步使用期间的存储生命周期。** 这不是让张量复制一份，也不是同步，而是帮助 PyTorch 缓存分配器知道这块内存还会被哪条流使用。若删除 `record_stream`，就必须以其他正确方式保证输入存储在使用完成前不被释放或复用。参见 PyTorch CUDA Semantics [3]。

最后的 `producer.wait_stream(stream)` 则为返回到原流后的消费建立依赖；脚本在计时和跨轮复用之前还会做一次设备同步。

这里补两个边界，免得把它当成万能模板：

- 本基准全程保留着输入张量的引用，最后又做了设备同步，所以 `x.record_stream(stream)` 主要起**教学和防御**作用——把“跨流使用时，存储生命周期要有人负责”这件事显式写出来。
- 如果输出要交给其他流异步消费，输出的存储生命周期同样要有人管。本基准是“最后统一同步、再校验结果”，因此它不能直接照搬为任意异步服务的完整写法。

**为什么不在每批后 `stream.synchronize()`？** 因为那样 CPU 每提交一批就等待这一批完成，原本可重叠的任务被人为排成串行。统一同步并不是唯一正确方法；Event 和流间等待也可以表达更细粒度的正确依赖。

### 这里有一个不容易发现的细节

同一个模型实例被多个 Stream 用于**只读推理**，本例不修改参数、不执行反向传播，没有显式共享的可变状态；这才构成独立批次计算的基础。`torch.inference_mode()` 关闭了自动求导并进一步减少相关开销，但不负责 Stream 依赖管理。

如果模型的 `forward` 内修改共享缓冲区、执行未受保护的就地写入，或者业务代码复用输出缓冲区，就不能简单照搬“多个流独立执行”的假设。

## 06｜怎样证明更快？别让计时骗了你

CUDA 运算常常采用异步提交。如果这样写：

```python
start = time.perf_counter()
outputs = run_batches(model, batches, streams, device)  # streams 已提前创建
print(time.perf_counter() - start)
```

计到的时间可能更接近**CPU 把任务交出去的时间**，并未完整覆盖 GPU 执行到完成的时间。

因此，我们采用这样的端到端墙钟计时（输入已位于显存，排除上传开销）：

```python
def timed_run(model, batches, streams, device, label=None):
    # label 只加在正式测量轮次，方便在时间线上定位区间
    scope = torch.cuda.nvtx.range(label) if label else nullcontext()
    with scope:
        torch.cuda.synchronize(device)  # 起点前清空未完成工作
        start = time.perf_counter()
        outputs = run_batches(model, batches, streams, device)
        torch.cuda.synchronize(device)  # 终点前确保 GPU 已完成
        return time.perf_counter() - start, outputs
```

这里的 `streams` 是准备阶段创建好、跨轮复用的同一组对象，创建成本不计入。计时边界覆盖「提交 → 依赖 → 前向 → 最终同步」，它包含 Python 提交开销和 GPU 计算耗时，适合比较这一小段推理工作负载的**墙钟完成时间**。不能把它等同于某个单独 Kernel 的纯设备执行时间；如果还要进一步拆解各 Kernel，就需要 CUDA Event 和性能分析工具。

### 为什么还要预热、重复多次？

第一次执行可能包含 CUDA 上下文初始化、库算法选择、内存分配和其他一次性成本。先对每种流配置预热，再重复测量，使用中位数而非一次结果，可降低偶发噪声的影响。

建议测试：

| 组别 | Stream 数量 | 输入 | 数据拷贝 | 观察目标 |
| --- | ---: | --- | --- | --- |
| A | 1 | 40 批 × 100 张 | 无 | 单流基线 |
| B | 2 | 同上 | 无 | 是否改善 |
| C | 4 | 同上 | 无 | 是否继续改善 |
| D | 8 | 同上 | 无 | 观察收益饱和或退化 |

固定 GPU、PyTorch/CUDA 版本、模型、Batch Size、数据类型、预热策略，尽量避免同时运行其他 GPU 负载。可进一步将 Batch Size 从 16、64、100、256 改变，检查趋势。

### 应该记录哪些数字？

吞吐量：

$$
\text{Throughput}=\frac{\text{完成的图像数}}{\text{本轮总耗时（秒）}}
$$

相对加速比：

$$
\text{Speedup}(k)=\frac{T_{\text{1 stream}}}{T_{\text{k streams}}}
$$

注意这是**一轮固定任务的完成时间加速比**，不是在线用户请求的 p95 延迟改善。如果生产系统用多流处理更多请求，排队和资源竞争仍可能使部分请求的延迟上升。

### 不要把先前的 1.96× 当成本篇结论

原实验曾报告单流 1.502 s、多流 0.768 s，约 1.96 倍。但两种实现同时改变了 Stream 与复制行为，且多流循环内存在阻塞式 `.cpu()`，缺少 GPU 时间线，因此不能将该数字直接归因于 Kernel 重叠。

本篇**不虚构新基准结果**：运行附带脚本后，将真实设备数据填入下表。

| Streams | 中位耗时（ms） | 图像/秒 | 相对单流加速 | 是否见到 Kernel 重叠 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 待实测 | 待实测 | 1.00× | 不适用 |
| 2 | 待实测 | 待实测 | 待实测 | 待分析 |
| 4 | 待实测 | 待实测 | 待实测 | 待分析 |
| 8 | 待实测 | 待实测 | 待实测 | 待分析 |

拿到「耗时」和「时间线」两组证据后，结论可以按下面的表来读：

| 实验现象 | 可以得出的判断 |
| --- | --- |
| 有 Kernel 重叠，总时间也缩短 | 当前配置的重叠伴随整体收益 |
| 有重叠，总时间却增长 | 重叠存在，但收益被争用或其他开销抵消 |
| 总时间缩短，未见明显重叠 | 不能将收益直接归因于 Kernel 并发，需继续定位 |
| 无重叠，也无加速 | 当前配置未表现出多流收益 |

有时测出来“多流没有明显加速”，也完全是有价值的实验结果。它意味着当前负载不一定存在可以被更多 Stream 利用的空闲执行资源，或其他开销抵消了收益。

## 07｜最关键的一步：打开 GPU 时间线

当代码打印“4 Streams 比 1 Stream 快”时，还差一个证据：GPU 在同一时间，是否真的跑了来自不同 Stream 的 Kernel？

*NVIDIA Nsight Systems* 可以采集 CPU CUDA API 调用和 GPU Kernel / memcpy 时间线，帮助区分提交与设备执行。官方用户指南说明了 CUDA workload trace 与不同 Stream 的时间线展示 [4]。

在安装好 Nsight Systems 的环境中，可以运行：

```bash
nsys profile --trace=cuda,nvtx \
  --output=cnn_stream_trace \
  python 31-cuda_stream_cnn_benchmark.py \
  --streams 1 4 --batches 40 --repeats 3
```

具体命令选项、权限及可用的硬件追踪模式应以当前 Nsight Systems 版本为准。

这条命令会把初始化、预热、正式测量和校验全程都采下来。为免被无关区间淹没，脚本用 NVTX 给每次正式测量打了标签（`streams=1`、`streams=4`……）。打开生成的 `.nsys-rep` 后，展开对应 GPU 的 **CUDA Stream** 轨道，只看这些 NVTX 区间即可。Nsight 官方也建议聚焦性能关键区间，并用 NVTX 标签把 CPU 活动和 GPU 工作关联起来。

还有一条纪律：**正式性能数字取自未开启 profiler 的运行**；时间线用于解释重叠与瓶颈，不要把两种运行条件下的耗时混在一起比较。

在工具中找两种信号：

```text
CPU 线程：
  cudaLaunchKernel ... cudaLaunchKernel ... cudaLaunchKernel ...
  └── 这是提交调用，不是 GPU 真正的 Kernel 执行时长

GPU Stream 0： [conv][relu][pool]......
GPU Stream 1：      [conv][relu][pool]......
                   ↑
           观察两条 GPU Kernel 区间是否真实交叠
```

**如果只看到 CPU launch 调用密集，并不能证明 GPU Kernel 在同时运行。** 如果不同 Stream 的 Kernel 时间区间在 GPU 时间线上明显重叠，才有更直接的并发证据。

此外还应观察 CPU 是否存在提交瓶颈、是否出现大范围空洞，以及多流是否反而增加了单批等待时间。

## 08｜为什么多流反而可能更慢？

假设单流时一个大 Batch 的卷积已经充分利用设备：

```text
一条流：
GPU 计算资源 |████████████████████████|

四条流：
Stream 0  ─┐
Stream 1  ─┤  →  仍是同一张 GPU，同一份有限资源
Stream 2  ─┤     更多待执行工作 ≠ 更多算力
Stream 3  ─┘
```

多流可能产生以下反效果：

1. **资源争抢**：各 Kernel 争用 SM、缓存与显存带宽。
2. **主机提交开销**：许多细粒度算子依赖 Python 或库调用；增加流并不能自动减少调用数量。
3. **额外内存占用**：多个活跃批次可能同时保留更多中间激活。
4. **延迟恶化**：总吞吐略有改善，也可能让单批次在设备上等待更久。
5. **模型或算子限制**：有些库运算的执行路径、工作空间需求或隐式依赖影响实际并发。

如果瓶颈已变成大量短 Kernel 的主机提交开销，后续可以研究 **CUDA Graphs**；如果瓶颈是批次太小，可以比较**增加 Batch Size**与**增加 Stream 数量**哪种更合适。两者都是值得探究的不同优化杠杆，不能预设哪一种更好。

## 09｜那本篇到底解决了什么？

回到园区视频分析的故事：

- 第一篇告诉我们：**业务任务能独立运行，不代表计算硬件一定能让它们并行。**
- 这一篇告诉我们：**CUDA Streams 允许表达独立的 GPU 工作序列，但真实重叠要通过执行时间线证明。**
- 下一篇将继续问：如果有十项推理任务和四张 GPU，**哪项任务该去哪里，才能让最后一张 GPU 尽量早完成？** 这就进入 List Scheduling、LPT 和动态负载均衡的世界。

### 读完自测：六个问题

1. Stream 和 GPU SM 有什么不同？
2. 同流有序、跨流可并发，究竟是什么意思？
3. 为什么四条 Stream 不等于四倍计算能力？
4. 为什么 `.cpu()` 可能破坏我们希望得到的并行提交节奏？
5. 为什么要用 `wait_stream()` / `record_stream()` 管理依赖与生命周期？
6. 为什么即使测出更快，也要打开 Nsight Systems 验证 Kernel 重叠？

真正理解 CUDA Streams，不是会写 `torch.cuda.Stream()`，而是能从任务依赖、硬件资源和时间线上解释：**某次并发为什么有效，另一次为什么无效。**

> **放到今天看：** 在更复杂的推理服务中，同样要区分任务依赖、提交开销和设备资源瓶颈。多流可以表达并发机会；如果瓶颈是大量短 Kernel 的提交开销，则可以进一步研究 CUDA Graphs。流内有序、跨流才有并发机会，最终仍由硬件资源和显存带宽说了算。

---

## 附录 A｜完整可运行实验

完整脚本见随文附件 **`31-cuda_stream_cnn_benchmark.py`**（正文只摘取最有教学价值的片段）。Python 3.10+、具备 CUDA 支持的 PyTorch 与 NVIDIA GPU 即可运行；不依赖 CIFAR-10 下载，也不执行训练。

脚本本身已经包含本文承诺的全部机制：

- `argparse` 参数解析（`--batches / --batch-size / --streams / --repeats`）；
- Streams 按配置创建一次并复用，创建成本不计入计时；
- 每种配置独立预热，再重复测量、交错顺序、取中位数（降低时钟/温度漂移与偶发噪声）；
- 每轮用 `torch.allclose` 对照单流参考输出；
- 输出跨轮管理：只保留单流参考输出，其余逐轮校验后即丢弃，避免堆积占用显存。

```bash
python 31-cuda_stream_cnn_benchmark.py
# 改变 batch 或 Stream 数量
python 31-cuda_stream_cnn_benchmark.py --batch-size 64 --streams 1 2 4 8
```

> 注意：脚本的主要任务是严谨对照，而不是保证加速。不同硬件、PyTorch/cuDNN 版本上的速度可能差别很大。初次运行也应查看显存余量。脚本不虚构任何数字，请以你本机的真实输出为准。

## 附录 B｜本文的边界与易错点

| 容易说错的话 | 更准确的说法 |
| --- | --- |
| “4 条流就能同时执行 4 个模型” | 本文是一个模型处理多个独立批次；多流并发须受硬件与依赖约束 |
| “`non_blocking=True` 一定发生传输与计算重叠” | 需要考虑页锁定内存、流、设备能力和实际操作顺序；属于下篇流水线专题 |
| “统一 `synchronize()` 就保证加速测量正确” | 还要统一工作负载、输入位置、内存传输口径、预热和计时边界 |
| “输出 `argmax` 相同就代表完全一致” | 分类标签相同不能证明 logits 数值一致；本实验用容差比较输出张量 |
| “多流提高吞吐，就降低每个请求延迟” | 总完成时间、吞吐与单请求延迟是不同指标 |
| “模型训练准确率 60% 说明多流推理正确” | 预测质量与并发计算正确性要分开验证 |

## 参考资料（2026 年 10 月核对）

[1] NVIDIA, *CUDA Programming Guide — Asynchronous Execution*：https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html

[2] NVIDIA, *CUDA C++ Best Practices Guide — Concurrent Kernel Execution*：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

[3] PyTorch, *CUDA Semantics — CUDA streams and synchronization*：https://docs.pytorch.org/docs/stable/notes/cuda

[4] NVIDIA, *Nsight Systems User Guide — CUDA Trace*：https://docs.nvidia.com/nsight-systems/UserGuide/index.html

> 说明：本篇强调可证伪、可复现的性能结论，不把前述示例中的 1.96 倍直接作为本稿的真实性能提升。本文没有在指定读者 GPU 上实测，因此性能表保留待测值。
