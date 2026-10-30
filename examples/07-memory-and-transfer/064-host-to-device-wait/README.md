# 064 · 数据还没送到，GPU 已经等急了

H2D 传输分两种内存：可分页内存（pageable）要先拷进驱动暂存区再上卡，
固定内存（pinned）可以被 DMA 直接读取。用错一种，GPU 就只能在原地等。

本篇把一次"从主机到设备"拆成四段计时：准备、提交、等待、总耗时，
对比三种来源：可分页、已固定、以及"临时 pin 再拷"。

## 前置

061（访存是瓶颈的前提）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `host_to_device_lab.py` | pageable / pinned / pin_then_copy 的分段耗时对比 | 博文代码块 #3 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python host_to_device_lab.py
```

脚本无参数，逐个数据量档位打印三段耗时，并校验拷贝结果的正确性。

## 环境要求

- 必须：NVIDIA GPU + CUDA 版 PyTorch
- 依赖层：`requirements/python-gpu.txt`
- 记录 pinned 内存占用：固定内存是锁页的物理内存，占用过大会影响整机

## 观察点

- 固定内存与可分页内存的传输时间差；小传输看延迟、大传输看带宽。
- 固定内存的收益来自免去一次内部暂存拷贝，不是「带宽更高」。

## 验证状态

**未实测。** 本机无 NVIDIA GPU，脚本在入口退出。

需要记录：三种来源的分段耗时、`pin_then_copy` 的 pin 开销是否被算进总耗时。
见 `expected_output.md`。

## 出处

- 博文：[64-数据还没送到GPU已经等急了.md](../../../blog/064-异步传输与CUDA Stream重叠.md)
- 来源代码块 #3，88 行，md5 `ade7b0e86d1a7ce5a3eb1e10ae955840`

代码与博文逐字一致，未改写。
