# 066 · 实测输出

**状态：未实测（需 2 张 NVIDIA GPU）。**

## 1. 环境

```
GPU 型号 ×2 / Compute Capability：
Driver / CUDA Toolkit：
torch / CUDA build：
拓扑（nvidia-smi topo -m 的结果）：
```

## 2. 运行结果

脚本先打印设备信息，再按"方向 × 载荷"分组打印两条路径：

| 项 | 待填值 |
|---|---|
| device 0 name / UUID | 待填 |
| device 1 name / UUID | 待填 |
| `can_device_access_peer(0,1)` | 待填（True/False） |
| `can_device_access_peer(1,0)` | 待填（True/False） |

每组（8 组）需要填两行：

| 方向 | 载荷 | 路径 | median (ms) | range (ms) | 有效速率 (GB/s) | 正确性 |
|---|---|---|---|---|---|---|
| 0→1 | 1 KiB | host_staged | 待填 | 待填 | 待填 | PASS |
| 0→1 | 1 KiB | framework_copy | 待填 | 待填 | 待填 | PASS |
| … | … | … | … | … | … | … |

判读方式：

- **小载荷（1 KiB）全部是延迟主导**，两条路径的差异反映的是"几次同步、几次驱动调用"，
  不是带宽。别用小载荷的 GB/s 下结论。
- **大载荷（64 MiB）才看带宽**。`framework_copy` 明显快于 `host_staged` 时，
  说明框架走了 P2P；两者接近，说明数据很可能绕了主机内存。
- **有效速率不是链路带宽**：它按"载荷字节 / 完成时间"算，含驱动与同步开销，
  永远低于理论峰值。想报链路带宽要用 `nvbandwidth` 这类专用工具。

## 3. 正确性为什么可信

每条路径跑之前会把 `target` 填成 NaN，再完整执行一次并做 `assert_close(rtol=0, atol=0)`。
这样"漏写目标"不会被上一次正确结果掩盖——NaN 会立刻让比较失败。

## 4. 这份数据不能说明什么

- **不能证明物理链路**：脚本自己打印了 `physical route unconfirmed`。
- **不是 NCCL 带宽**：这是点对点拷贝，不是集合通信。
- **单机双卡**：跨节点要走网络，结论完全不同。
