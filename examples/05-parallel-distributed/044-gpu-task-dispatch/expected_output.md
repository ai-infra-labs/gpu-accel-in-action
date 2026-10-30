# 044 · 实测输出

**状态：已实测通过**（CPU，2026-10-28）。

## 1. 环境

```
平台    ：macOS 27.0.1 (arm64, Apple M4)
Python  ：3.14.3
依赖    ：无（仅标准库 dataclasses）
```

脚本不访问 GPU、不加载模型，输出与显卡型号无关。

## 2. 运行结果

退出码 `0`，原始输出：

```
空闲但没加载目标模型
shortest_queue: identity@v1/GPU0/instance0, 预计完成 20 ms
earliest_finish: identity@v1/GPU0/instance0, 预计完成 20 ms

空闲但数据准备更慢
shortest_queue: identity@v1/GPU1/instance0, 预计完成 31 ms
earliest_finish: identity@v1/GPU0/instance0, 预计完成 20 ms

队伍短但前面的工作更重
shortest_queue: identity@v1/GPU0/instance0, 预计完成 88 ms
earliest_finish: identity@v1/GPU1/instance0, 预计完成 32 ms
```

| 场景 | `shortest_queue` | `earliest_finish` | 差在哪 |
|---|---|---|---|
| 空闲但没加载目标模型 | GPU0，20 ms | GPU0，20 ms | 两种策略结论相同：GPU1 的 `models` 为空，先被候选过滤掉 |
| 空闲但数据准备更慢 | GPU1，31 ms | GPU0，20 ms | 队列短 ≠ 完成早：GPU1 要等 25 ms 数据就绪 |
| 队伍短但前面的工作更重 | GPU0，88 ms | GPU1，32 ms | 队列 1 条但前面的任务要跑 80 ms |

## 3. 这份数据能说明什么

- **候选过滤先于策略**：策略只在 `ready == True` 且 `model in models` 的实例里挑。
  第一组两种策略同选 GPU0，不是因为 GPU1 远，而是因为它根本不在候选里。
- **"最空闲"是一种代理指标**：它的假设是"排队少 ⇒ 完成早"，这个假设在第二、
  第三组里都不成立。
- **`earliest_finish` 也不是免费的**：它需要每次预估 `compute_ms`，估错了照样选错。
  本篇没有验证预估误差的影响。

## 4. 这份数据不能说明什么

- 不是硬件实测：所有时间都是代码里的教学假设，不构成性能承诺。
- 不涉及网络排队、抢占、实例伸缩，也不涉及真实的模型加载耗时。
- 6 行输出只是三个固定快照，不是调度器在真实负载下的行为。
