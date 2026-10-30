# 两张 GPU 传数据，真的走了 NVLink 吗？

> **GPU 内存优化 · 第 7 篇**  
> 一行设备复制说明了数据的起点和终点，却没有交代完整路线。先核对连接与能力，再讨论速度。

> **工程深度｜验证状态：** L2–L3 工程｜概念验证（拓扑需实测）
假设你把一段 AI 处理流程放在 GPU 0，另一段放在 GPU 1。前一段产生特征，后一段接着计算。连接它们的代码很短：

```python
features_1 = features_0.to("cuda:1", non_blocking=True)
```

有人看到这行代码，就说：“这是 GPU 直传，走 NVLink，比经过 CPU 快。”

这句话至少跳过了两个问题：两张卡之间真的有 NVLink 吗？本次复制是否使用了这条连接？

前两篇讨论 CPU–GPU 传输，以及怎样用依赖与双缓冲组织重叠。本篇转向两张 GPU 之间的搬运，围绕一个问题收束系列：**怎样把传输路径和带宽测清楚，避免为一条未经证实的路线命名？**

下面的流水线是教学场景。本文没有双 GPU 实测记录，不提供虚构的拓扑、时间或加速比；测量代码尚未完成 GPU 执行验证。资料核对日期：2026 年 10 月 20 日。

## 01｜“GPU 到 GPU”，不是一种固定的物理路线

先区分三件事：

| 层面 | 回答的问题 | 例子 |
|---|---|---|
| 应用意图 | 从哪里复制到哪里？ | GPU 0 的张量复制到 GPU 1 |
| 访问与复制能力 | 两个设备能否使用对等访问或设备间复制？ | CUDA P2P 能力与相应接口 |
| 物理互联 | 数据经过哪些硬件连接？ | PCIe、NVLink，以及相关交换结构 |

**P2P 是 peer-to-peer，对等访问。**在 CUDA 中，设备能否访问另一设备的内存，与系统拓扑及支持情况有关。官方编程指南明确将 PCIe 和 NVLink 都列为相关拓扑条件。[1] 下图是整机视角下 CPU、GPU 与互联通道的位置——后面反复提到的"物理路径"，指的就是这些通道中实际走的是哪一条：

![CPU、GPU 与 PCIe/NVLink 互联的整机示意](assets/figure-cpu-gpu-interconnect.jpg)

所以，对等访问不等于 NVLink。支持的系统可以通过 PCIe 实现 GPU 之间的对等传输，不必先把数据放入应用创建的 CPU 中转张量。

NVLink 是另一类互联技术，某些系统还使用 NVSwitch 组织连接。是否具备这些连接，要看具体 GPU、服务器结构与配置，不能仅凭“两张 NVIDIA 显卡”推断。

再补一个边界：**数据经过 CPU 所在的 PCIe 主桥，不等于数据一定先落入主机内存中转。**拓扑中有 CPU 相关部件，和应用显式执行 GPU→CPU→GPU，是两回事。

## 02｜两种常见称呼，实际比较了什么

一种常见的错误比较，是把显式主机中转称为“PCIe”，把框架设备间复制称为“NVLink”：

| 被这样称呼的操作 | 更准确的名称 |
|---|---|
| GPU 0 → CPU 张量 → GPU 1 | 显式主机中转 |
| GPU 0 的张量直接调用设备复制 API 到 GPU 1 | 框架设备间复制，物理路径待核对 |

第一条路线明确经过了应用创建的主机缓冲区，但不应该被当作所有 PCIe 设备间传输的代表。第二条没有显式创建主机中转对象，也不能仅凭源码命名为 NVLink。

CUDA 文档分别讨论对等内存访问与设备间复制；在满足并启用对等访问的条件下，设备间复制可以避免主机暂存。[1] 但框架在具体环境中采用什么实现，还取决于其后端与平台支持，不能反过来从一行高层 API 推断所有细节。

本篇因此只使用两个实验标签：

- `host_staged`：我们明确安排的主机中转路线。
- `framework_copy`：交给 PyTorch 的设备间复制请求。

**路径尚未确认，是一个有效结论。**它比把“猜测是 NVLink”写成性能事实更有用。

## 03｜测之前，先确认两张卡是谁、怎样连接

先记录型号、驱动与设备身份，再看拓扑。可以在支持 NVIDIA 管理工具的环境运行：

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version --format=csv
nvidia-smi topo -m
```

第一条帮助识别 GPU，第二条展示连接关系。输出中的缩写需要按工具图例解读，例如：[2]

| 标记 | 大致说明什么 |
|---|---|
| `PIX`、`PXB` | 经过不同层级的 PCIe 桥或交换结构 |
| `PHB` | 路径涉及 PCIe 主桥 |
| `NODE`、`SYS` | 涉及 NUMA 节点内部或跨节点的连接关系 |
| `NV#` | 工具识别到由若干 NVLink 组成的连接 |

这张表用于读路线，不用于排列绝对速度。具体定义以当前工具输出为准；同一种标记下，实际性能也可能不同。

还可以检查能力与连接状态：

```bash
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
nvidia-smi topo -p2p n
nvidia-smi nvlink --status
```

`r`、`w`、`n` 分别查询 P2P 读、写和 NVLink 相关能力；NVLink 状态查询用于查看连接状态。[2] 选项或硬件不支持时，应记录“不支持该查询”，不能把缺失信息当作带宽为零。

### PyTorch 里的设备 0，不一定是管理工具里的设备 0

容器可见设备范围、`CUDA_VISIBLE_DEVICES` 和枚举顺序，可能改变程序中的逻辑编号。CUDA 官方文档说明，可见设备会按指定顺序重新枚举。[1]

因此，多个相同型号 GPU 尤其要根据 UUID、PCI 总线地址等信息对应身份，不要仅按名称或编号猜测。

在 PyTorch 中，还可以查询：

```python
print(torch.cuda.can_device_access_peer(0, 1))
print(torch.cuda.can_device_access_peer(1, 0))
```

这个 API 回答“一个设备能否对等访问另一个设备”，不是返回“当前复制正在走哪条物理链路”。两个方向也应分别记录。[3]

到这里，我们得到的是**硬件连接与能力证据**。要确认实际流量，还需要结合测试配置与观测，而不是看到一个 `True` 就结束。

## 04｜先把起点和终点固定，再量完成时间

我们要比较的是：同一份 GPU 数据，何时完整抵达另一张 GPU。

为了减少干扰，先固定条件：

- 输入是源 GPU 上的连续 FP32 张量。
- 目标 GPU 的输出提前分配，避免把每次输出分配混入测试。
- 主机中转缓冲区也提前分配，使用页锁定内存。
- 不转换形状、布局与数据类型。
- 计时前排空两张卡上的相关工作，计时结束前再次等待两张卡。

显式主机中转分成两步：先完成源 GPU→主机，再启动主机→目标 GPU。这里故意保留完成等待，保证 CPU 缓冲区的数据已经就绪，不测两段传输流水化。

框架复制则调用目标张量的 `copy_()`。PyTorch 允许源与目标处于不同设备。[4] 实验标签描述的是 API 层的请求，不保证没有任何内部暂存，也不保证使用了 NVLink。

### 为什么不只同步一次“当前设备”

不带设备参数的 `torch.cuda.synchronize()` 等待当前设备。[5] 多 GPU 代码中，当前设备可能不是你以为的那个。

如果只等待源卡，目标卡工作是否完成可能没有覆盖；若连初始化或输入生成也未排除，时间还可能混入上一段任务。

本篇采用保守、容易审阅的口径：**两端都明确同步，墙钟计时覆盖复制请求到两端完成。**它包含 Python 调用、上下文操作与同步开销，不是单独测量硬件传输引擎的纯工作时间。

也不要在源卡、目标卡分别记录一个 CUDA Event，再直接相减。CUDA 的事件耗时计算要求两个事件处于同一设备；跨设备时间线需要另行组织。[1][6]

## 05｜带宽数字，最容易在哪里说错

设本次交付给目标 GPU 的有效数据量是 S 字节，从启动复制到完成用了 t 秒，本实验报告：

\[
B_{\text{交付}}=\frac{S}{t\times10^9}\ \text{GB/s}
\]

这是**单向、按有效载荷计算的交付速率**，包含上述等待开销。

主机中转虽然执行了两段复制，但目标卡最终收到的仍是 S 字节，所以该列分子仍用 S。若改用 2S，只能另称“两段复制累计字节的统计口径”，不能与直接交付的 S/t 混在同一列。

对两段完全串行的中转，可以写一个教学近似：

\[
T_{\text{中转}}\approx
L_1+\frac{S}{B_1}+L_2+\frac{S}{B_2}
\]

其中 L 代表固定开销，B 代表各段的有效传输能力。这个模型帮助解释“两段相加”的成本，不预测具体设备成绩；也没有覆盖分块流水化的中转实现。

比较其他资料时，还要对齐四组口径：

| 容易混淆的数字 | 需要核对 |
|---|---|
| GB/s 与 GiB/s | 除数分别是十进制与二进制单位 |
| 单向与双向 | 是否同时统计两个方向的总量 |
| 单条连接与整卡聚合 | 是否累加多条链路或交换结构能力 |
| 理论规格与实测 | 是否包括协议、软件、同步和应用行为 |

本文不列一个脱离型号与代际的“NVLink 标准带宽”，也不把厂商双向聚合规格直接拿来评判一次单向张量复制。

数据大小同样重要：小消息更容易被启动和同步固定成本主导，大块数据才更适合观察持续搬运能力。即使如此，重复读写同一缓冲区也会影响缓存状态，不能未经检查就把结果等同于物理显存或互联峰值。

## 06｜完整实验：两种调用方式，两个传输方向

**环境与验证状态：**需要至少两张对进程可见的 CUDA GPU、兼容驱动和 CUDA 版 PyTorch。脚本按 PyTorch 2.x 常用公开接口编写，在单进程、默认 stream、连续 FP32 数组条件下测量，不使用跨样本重叠。本生成环境未安装 PyTorch，代码仅经过 Python 语法检查和静态逻辑审阅，尚未完成 GPU 执行、路径确认或性能验证。

保存为 `gpu_pair_transfer_lab.py`，执行 `python gpu_pair_transfer_lab.py`。它选逻辑设备 0 和 1；如果身份映射不清楚，先完成前面的核对，不要将结果套到另一对卡。

```python
"""两张 GPU 传数据，真的走了 NVLink 吗？（第7章 · 访存与传输）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch。
"""
import time
from statistics import median
import torch

ROUTES = ("host_staged", "framework_copy")


def synchronize_pair(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def one_sample(route, source, target, staging, devices):
    synchronize_pair(devices)
    start = time.perf_counter()
    if route == "host_staged":
        with torch.cuda.device(source.device):
            staging.copy_(source, non_blocking=True)
        # CPU 缓冲区完成接收后，才允许目标 GPU 从中读取。
        torch.cuda.synchronize(source.device)
        with torch.cuda.device(target.device):
            target.copy_(staging, non_blocking=True)
    elif route == "framework_copy":
        with torch.cuda.device(target.device):
            target.copy_(source, non_blocking=True)
    else:
        raise ValueError(route)
    synchronize_pair(devices)
    return (time.perf_counter() - start) * 1000


def run_case(source_id, target_id, payload_bytes):
    source_device = torch.device(f"cuda:{source_id}")
    target_device = torch.device(f"cuda:{target_id}")
    devices = (source_device, target_device)
    elements = payload_bytes // 4
    source = torch.randn(elements, dtype=torch.float32, device=source_device)
    target = torch.empty(elements, dtype=torch.float32, device=target_device)
    staging = torch.empty(
        elements, dtype=torch.float32, device="cpu", pin_memory=True
    )
    synchronize_pair(devices)
    reference = source.cpu()  # 正确性参照，放在计时外。
    print(f"\n{source_id} -> {target_id}, payload={payload_bytes} bytes")

    for route in ROUTES:
        for _ in range(3):
            one_sample(route, source, target, staging, devices)
        # 清成 NaN 后再完整执行一次：若本路线漏写目标，
        # 比较会失败，而不会被之前路线留下的正确旧结果掩盖。
        target.fill_(float("nan"))
        one_sample(route, source, target, staging, devices)
        torch.testing.assert_close(target.cpu(), reference, rtol=0, atol=0)
        print(f"  {route}: correctness PASS")

    records = {route: [] for route in ROUTES}
    for round_id in range(8):
        order = ROUTES if round_id % 2 == 0 else ROUTES[::-1]
        for route in order:
            records[route].append(
                one_sample(route, source, target, staging, devices)
            )
    for route in ROUTES:
        samples = records[route]
        mid = median(samples)
        if mid <= 0:
            raise RuntimeError("计时分辨率不足，无法计算有效速率。")
        gbps = payload_bytes / (mid / 1000) / 1e9
        print(
            f"  {route}: completion median={mid:.6f} ms"
            f", range=[{min(samples):.6f}, {max(samples):.6f}] ms"
            f", payload delivery={gbps:.3f} GB/s"
        )


@torch.no_grad()
def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("需要至少两张可见的 CUDA GPU。")
    torch.manual_seed(42)
    print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
    for device_id in (0, 1):
        props = torch.cuda.get_device_properties(device_id)
        print(
            "logical device:", device_id, "name:", props.name,
            "UUID:", getattr(props, "uuid", "本版本未提供此属性")
        )
    print("device 0 can access device 1:", torch.cuda.can_device_access_peer(0, 1))
    print("device 1 can access device 0:", torch.cuda.can_device_access_peer(1, 0))
    print("warmup=3 per route, samples=8 per route; physical route unconfirmed")
    for source_id, target_id in ((0, 1), (1, 0)):
        for payload_bytes in (1 << 10, 1 << 20, 16 << 20, 64 << 20):
            run_case(source_id, target_id, payload_bytes)


if __name__ == "__main__":
    main()
```

最大的载荷是 64 MiB，还会保留两端 GPU 数组、页锁定中转、CPU 参照及正确性检查副本，另有运行时和缓存开销。内存紧张可缩小规模。

如果 P2P 能力查询为假，不应自行将框架复制命名为“PCIe P2P”。即便高层复制成功，其底层路径仍需核对；若调用失败，应保留错误，并报告该环境未完成此项测量，不把失败写成零带宽或悄悄替换测试后继续沿用原标签。

## 07｜用官方工具交叉核对，别让一个脚本承担所有结论

上面的脚本擅长回答应用层问题：我这样提交复制，等待数据到达，需要多久？

如果要进一步评估硬件与底层传输能力，可以使用 NVIDIA 的 **nvbandwidth**。官方项目提供设备间、主机与设备间等测试，并区分不同测试方式。[7]

按官方说明构建或安装后，先查看版本与可用测试：

```bash
./nvbandwidth --version
./nvbandwidth -l
```

若当前版本包含官方列出的设备间复制测试，可进一步运行：

```bash
./nvbandwidth -b 64 -t device_to_device_memcpy_write_ce
```

其中 `-b` 的缓冲区单位是 MiB；实际可用测试和参数以本地帮助及版本为准。建议记录工具版本或提交号，避免项目更新后仅凭记忆无法复现。[7]

输出的设备矩阵有明确的方向定义：对官方示例中的 `device_to_device_memcpy_write_ce`，表头是 **`GPU(row) ← GPU(column)`，即列是源、行是目标**。对照本脚本的 `0→1` 时，应查看行 1、列 0。名字里的 **CE** 表示使用复制引擎（Copy Engine）的测试；工具还包含由计算内核搬运数据的测试，两者不要混为同一个基线。

多卡系统还应按工具支持的方式选择目标设备对。当前官方项目提供按 UUID 限定设备对的选项（以本地版本支持为准）：

```bash
# 仅当本地版本支持 --pair 时使用；UUID 以实际查询结果替换
./nvbandwidth -b 64 -t device_to_device_memcpy_write_ce \
  --pair GPU-<UUID-A> GPU-<UUID-B>
```

尤其不要默认“测试名称带双向，输出就是两个方向相加”。nvbandwidth 当前说明中的双向结果，表示存在反向并发流量时的每方向带宽，而不是聚合总和；比较前仍需核对所用版本的定义。[7]

不要要求它与本文墙钟脚本给出完全相同的数值：测试实现、重复策略和同步口径并不相同。显式主机中转还可能受到 CPU／NUMA 亲和性及主机内存位置的影响——nvbandwidth 提供自动 CPU 亲和性控制，其环境安排可能与 Python 脚本不同。这里只解释差异来源，不要求读者修改系统配置。二者差距明显时，应检查差异来源，而不是挑更大的数字作为结论。

### 怎样才更有把握说“使用了 NVLink”

至少把证据分成三层：

| 证据 | 能支持什么 | 尚不能单独证明什么 |
|---|---|---|
| 拓扑与链路状态 | 这对设备之间具备相关连接 | 每次框架复制都使用它 |
| P2P 能力与测试配置 | 当前环境支持相关能力，测试如何提交 | 一行高层 API 的完整内部路径 |
| 隔离测试期间的链路流量观测 | 流量与测试在时间、方向上相符 | 存在其他流量时，全部增长都来自某一条 API |

工具和硬件支持时，可以结合相关链路的吞吐量监测进一步核对，使用当前 `nvidia-smi nvlink -h` 所列的观测能力，不假设所有代际都提供相同计数器。[2]

链路流量核对可以按一个短流程执行：

1. 先记录空闲时的链路流量基线。
2. 只运行指定设备对、指定方向的持续复制。
3. 比较测试期间相关端点的发送与接收变化。
4. 停止复制后再次观察，确认流量随之回落。
5. 将流量变化与测试时间对应起来，并排除其他任务干扰。

两点边界要说清楚：**单次短复制可能被监测采样间隔掩盖**，所以路径观测可以另设一个持续复制的测试，不必改变本文现有的测速口径；而流量观测通常只能支持“这段测试使用了相关连接”，不能轻易把每一笔增长精确归因到某一次 Python 调用。具体计数器名称与用法，以硬件和当前工具帮助为准。

如果只能确认“拓扑有 NVLink、复制成功”，就如实写到这一层。更快的成绩也不是链路身份识别器。

## 08｜快一点的互联，什么时候能改善 AI 任务

设一段原本串行的教学任务，由计算时间 C 与传输时间 T 组成。即使传输被完全消除，总时间也最多从 C+T 降到 C。因此，通信原本占多少，决定了优化它的空间。

如果原流程已经隐藏了部分通信等待，更快链路的应用收益还需要重新测量，不能把孤立复制的提升直接乘到整段推理或训练上。

实践中，值得先看四件事：

- **传什么**：是否传了下游不需要的中间结果？
- **传多大、传几次**：大量小消息可能更多受提交和同步成本影响。
- **什么时候等**：复制是否真的落在关键路径上，是否存在无谓的全局等待？
- **设备怎么放**：若有多组 GPU 可选，任务所用的设备对是否匹配合适拓扑？

本篇不展开张量并行、流水线并行或集合通信算法。点对点复制只是其中的基础观察，不能代表 AllReduce 等完整通信模式的性能。

最后，把七篇的判断方法放回同一张表：

| 遇到的现象 | 优先追问 |
|---|---|
| 删了张量，显存没归零 | 哪些对象还活着，哪些只是缓存？ |
| 计算简单，内核仍慢 | 相邻线程在同一条指令上访问哪些地址？ |
| 使用共享内存却没变快 | 省下什么，又增加了哪些同步和资源成本？ |
| 只读系数表没有优势 | warp 内是否读取相同地址？ |
| GPU 在等 CPU 送数据 | 有没有多余往返，准备成本算进去了吗？ |
| 打开异步仍没加速 | 时间线上真的重叠了吗，依赖是否正确？ |
| 两卡交换不够快 | 身份、拓扑、路径、方向与测量口径是否对齐？ |

GPU 内存优化最终需要一条可核查的解释：数据放在哪里，谁在何时使用它，哪些工作因此减少，结果怎样验证。

面对两张 GPU，先把复制叫作你真正测到的东西。能确认主机中转，就写主机中转；只确认框架复制，就保留路径未知。等证据把路线补齐，再讨论它是不是你期待的 NVLink。

## 参考资料

以下优先使用厂商和项目官方资料。CUDA、PyTorch 固定版本链接用于明确核验口径，不表示它们是当前最新版本；管理工具与 nvbandwidth 的选项应以实际安装版本为准。

1. [NVIDIA：CUDA C++ Programming Guide 12.8](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html) — Peer-to-Peer Memory Access、Peer-to-Peer Memory Copy 与设备枚举环境变量。
2. [NVIDIA：NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html) — GPU 身份、拓扑图例、P2P 能力、NVLink 状态及观测选项。
3. [PyTorch：torch.cuda.can_device_access_peer](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.can_device_access_peer.html) — 查询设备对等访问能力。
4. [PyTorch：Tensor.copy_](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.copy_.html) 与 [CUDA semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html) — 跨设备复制与执行语义。
5. [PyTorch：torch.cuda.synchronize](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.synchronize.html) — 等待指定设备上的 CUDA 工作完成。
6. [NVIDIA：CUDA Runtime API — Event Management](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__EVENT.html) 与 [PyTorch：torch.cuda.Event](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.Event.html) — 事件设备约束与耗时计算。
7. [NVIDIA：nvbandwidth 官方项目](https://github.com/NVIDIA/nvbandwidth) — 测试方式、缓冲区单位、设备间带宽测试与输出方向。

图片来源：第 01 节的整机互联示意图为教学示意图，展示 CPU、GPU 与 PCIe/NVLink 通道的位置关系；具体系统中哪些 GPU 具备 NVLink、数据实际走哪条通道，仍以 `nvidia-smi topo -m` 等实测输出为准。

材料基础：用户提供的《第 8 章 GPU 内存优化》图片文字转写稿。本篇重构其 PCIe 与 NVLink 示例，区分显式主机中转、框架复制与物理互联，修正同步及带宽口径，未沿用未经核验的运行时间和加速比。
