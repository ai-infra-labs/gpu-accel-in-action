# 042 · 开了四个线程，GPU 就会同时跑吗？

把工作拆到多个 CUDA Stream 上，不等于它们真的并发。并发需要三个条件同时满足：
Stream 之间没有依赖、GPU 有剩余资源、计时口径能看到重叠。

本篇用同一批模型，对比串行执行与多 Stream 并行执行，同时打印
Event 区间时间与墙钟时间——两者的差值就是"是否真的重叠"的证据。

## 前置

003（CPU 与 GPU 的并发编排）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `streams_demo.py` | 串行 vs 多 Stream 的 Event/墙钟双口径对比 | 博文代码块 #8 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python streams_demo.py
```

脚本无参数，直接打印 GPU 型号、PyTorch/CUDA 版本、每种调度的两种计时，
并校验两种调度的输出是否一致。

## 环境要求

- 必须：NVIDIA GPU + CUDA 版 PyTorch（脚本开头有 `torch.cuda.is_available()` 检查，
  不满足时直接 `SystemExit`）
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- 多 Stream 是否真并发，看总时间接近「单流」还是「4 倍串行」，不看单次调用快慢。
- 没变快时先查有没有在循环里 synchronize，以及工作是否真的互不依赖。

## 验证状态

**未实测。** 本机无 NVIDIA GPU，`torch.cuda.is_available()` 为 False，
脚本会在入口直接退出。

需要记录：每种调度的 Event 时间与墙钟时间、两者比值、输出校验结果。见 `expected_output.md`。

## 出处

- 博文：[42-开了四个线程GPU就会同时跑吗.md](../../../blog/042-开了四个线程GPU就会同时跑吗.md)
- 来源代码块 #8，96 行，md5 `27481e89bce41e6f3c09d03da1cfe543`

代码与博文逐字一致，未改写。
