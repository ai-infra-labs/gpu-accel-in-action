# 065 · 打开异步传输，为什么还是没变快？

`non_blocking=True` 只是允许提交立即返回，**不保证**它和计算真的重叠。
重叠还需要：独立的拷贝 Stream、主机内存已固定、以及正确的 Event 同步顺序。

本篇在三种模式下跑同一条"拷贝 + 计算"流水线：全阻塞、同 Stream 异步、
双 Stream 重叠，逐模式对比总耗时，并输出 trace 文件供可视化。

## 前置

064（传输方式）、042（并发）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `overlap_lab.py` | blocking / same_stream / overlap 三模式对比 | 博文代码块 #2 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python overlap_lab.py
```

脚本无参数。三种模式运行完后校验结果是否与 blocking 基线一致，
并写出 trace 文件路径（供时间线分析）。

## 环境要求

- 必须：NVIDIA GPU + CUDA 版 PyTorch
- 依赖层：`requirements/python-gpu.txt`
- 双 Stream 重叠需要主机内存已 pin，否则拷贝退化为同步

## 观察点

- 异步是否真重叠：总时间应接近 max(拷贝, 计算)，而不是两者之和。
- 没变快时先查有没有在拷贝与计算之间插了同步，或者工作区不可并发。

## 验证状态

**未实测。** 本机无 NVIDIA GPU，脚本在入口退出。

需要记录：三种模式总耗时、`overlap` 相对 `blocking` 的比值、
结果一致性、trace 文件里是否真的看到拷贝与计算重叠。见 `expected_output.md`。

## 出处

- 博文：[65-打开异步传输为什么还是没变快.md](../../../blog/065-打开异步传输为什么还是没变快.md)
- 来源代码块 #2，123 行，md5 `460c50482420ddc87f2a3ae3754af786`

代码与博文逐字一致，未改写。
