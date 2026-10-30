# 010 · 实测输出

**状态：未实测。**

本机（Apple Silicon，无 `nvcc`、无 NVIDIA GPU）无法编译或运行 PyCUDA，
以下模板为待填字段。**不得用推断值填充**，任何数字都必须来自真实运行。

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
PyCUDA / NumPy 版本：
block_size：
```

## 2. 正确性

| 项 | 结果 |
|---|---|
| `--test quick` | 待填（通过 / 失败用例） |
| `--test full` | 待填（通过 / 失败用例） |
| N=1 不变量自检 | 待填 |
| Kernel 计数自检 | 待填 |
| 计时后复核结果 | 待填（是 / 否） |

`--test quick` 只确认环境与基本路径可用，**不能**替代 `full`：后者才覆盖
递归 scan 与多组 block 配置。

## 3. Pipeline 成本

| 路径 | Kernel 启动次数 | 中间缓冲区个数 |
|---|---|---|
| Reduction | 待填 | 待填 |
| Scan | 待填 | 待填 |

博文给出的参考值（3/3 与 5/6）需用运行输出核对后再确认，不要直接抄。

## 4. 性能

计时口径：GPU Event 区间（不含 D2H 回传），同一 Stream 空闲、无其他 GPU 负载。

| 路径 | warmup | iterations | median (ms) | P95 (ms) | min (ms) |
|---|---|---|---|---|---|
| Reduction | 待填 | 待填 | 待填 | 待填 | 待填 |
| Scan | 待填 | 待填 | 待填 | 待填 | 待填 |

CPU 基线（NumPy）：

| 精度 | median (ms) | Speedup |
|---|---|---|
| float32 | 待填 | 待填 |
| float64 | 待填 | 待填 |

## 5. 踩坑记录

待填。至少记录：

- 首次运行的失败信息与真实原因
- `block_size` 取值对 shared memory 上限的影响
- shared memory 与 occupancy 的实际表现（如采集）
