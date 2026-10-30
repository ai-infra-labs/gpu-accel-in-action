# 066 · 两张 GPU 传数据，真的走了 NVLink 吗？

`tensor.to("cuda:1")` 只说明"数据到了另一张卡"，不说明走了哪条路：
可能是 P2P 直连（NVLink / PCIe P2P），也可能是先回主机内存再过去。
本篇用两种写法对照——显式 host staging 与框架侧 copy——看完成时间差多少。

## 前置

064、065（传输与重叠）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `nvlink_copy_bench.py` | 两条拷贝路径 × 2 个方向 × 4 种载荷，正确性 + 完成时间 | 博文代码块 #4 |

## 运行

```bash
python -m pip install -r ../../requirements/advanced.txt
python nvlink_copy_bench.py
```

必须有 2 张可见 GPU。总样本 2 方向 × 4 载荷 × 2 路径 × 8 次 = 128 次拷贝。

## 环境要求

- 2 张 NVIDIA GPU（脚本检查 `device_count() >= 2`）
- 依赖层：`requirements/advanced.txt`

## 观察点

- 大载荷（64 MiB）看带宽：framework_copy 明显快于 host_staged 才说明走了 P2P。
- 小载荷全是延迟主导，此时的 GB/s 不能当链路带宽用。

## 验证状态

**未实测。** 本机无 GPU，仅完成语法检查。

需要记录：两张卡的 UUID 与 peer access 结果、每条路径的 median/range/GB/s。
见 `expected_output.md`。

## 为什么不是"NVLink 检测工具"

脚本打印的是**完成时间与有效速率**，不是物理链路。要确认链路，得用
`nvidia-smi topo -m` 或 NVML 的 `nvmlDeviceGetP2PStatus`——本脚本刻意不去猜。
`can_device_access_peer` 为真只说明"允许 P2P"，不等于"这次真的走了 P2P"。

## 出处

- 博文：[66-两张GPU传数据真的走了NVLink吗.md](../../../blog/066-两张GPU传数据真的走了NVLink吗.md)
- 来源代码块 #4，97 行，md5 `feaccbc6a58cfceb3ac2b40224c8dbc8`

代码逐字一致。**博文未指定文件名**，`nvlink_copy_bench.py` 为仓库新增命名。
