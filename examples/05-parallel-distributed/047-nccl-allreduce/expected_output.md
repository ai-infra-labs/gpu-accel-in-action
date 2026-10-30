# 047 · 实测输出

**状态：未实测（需 2 张 NVIDIA GPU + torchrun）。**

## 1. 环境

```
GPU 型号 ×2 / Compute Capability：
Driver / CUDA Toolkit：
torch / NCCL 版本：
启动命令：torchrun --nproc_per_node=2 nccl_allreduce_torchrun.py
```

## 2. 预期输出（由代码里的常量推得，不是实测）

rank 0 会打印四行：

```
AllReduce: [11.0, 22.0, 33.0, 44.0]
原始输入 AllGather: [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
ReduceScatter 后再 AllGather: [11.0, 22.0, 33.0, 44.0]
结果校验通过；本例不测试通信速度。
```

| 原语 | 输入 | 输出 | 语义 |
|---|---|---|---|
| AllReduce(SUM) | rank0 `[1,2,3,4]`、rank1 `[10,20,30,40]` | 每个 rank 都是 `[11,22,33,44]` | 逐元素求和，结果**复制到所有 rank** |
| AllGather | 各自的 `[1,2,3,4]` / `[10,20,30,40]` | 每个 rank 都是 8 个元素 | 按 rank 顺序拼接，**不做运算** |
| ReduceScatter(SUM) | 同上，长度 4 | 每个 rank 拿 2 个元素 | 先求和再按 rank 切分 |

把第三行的结果在 rank 间拼起来，正好等于第一行——这条等式就是
"ReduceScatter + AllGather ≡ AllReduce" 的证据，也是 Ring AllReduce 的基本单元。

三组 `torch.testing.assert_close` 全过才会打印上面四行。打印不出来 = 校验没过，
不要只看有没有报错。

## 3. 这份数据不能说明什么

- **不是带宽测试**：4 个 float32 的通信量，测出来全是启动延迟。
- **不涉及 NVLink 是否生效**：本篇只验证语义；走 PCIe 还是 NVLink，
  结果一模一样，只能从耗时上看出来（见 066）。
- **没有梯度、没有模型**：这是通信原语的语法课，不是 DDP 实战。
