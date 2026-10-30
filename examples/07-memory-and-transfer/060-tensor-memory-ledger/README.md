# 060 · 删了张量，显存为什么还没降？

`del x` 只是减引用计数，不等于把显存还给驱动。中间结果、计算图、缓存块、
分配器保留的空闲块，任何一层没释放，`nvidia-smi` 上的数字就不会动。

本篇按阶段记账：每步之后打印当前已分配 / 已保留 / 峰值显存，把"看不见的占用"
变成可对照的数字。

## 前置

043（显存争用的现象）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `memory_ledger.py` | 分阶段显存记账 + `gc` 回收对照 | 博文代码块 #5 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python memory_ledger.py
```

脚本打印 PyTorch 版本、CUDA build、GPU 名称、**分配器后端**，然后按轮次
输出各阶段显存数字。入口会检查 `torch.cuda.is_available()`。

## 环境要求

- 必须：NVIDIA GPU + CUDA 版 PyTorch
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- del 之后显存是否立刻下降：allocator 缓存与真正归还驱动是两件事。
- 用 memory_allocated 与 memory_reserved 两个数一起看，差值就是缓存。

## 验证状态

**未实测。** 本机无 NVIDIA GPU，脚本在入口退出。

需要记录：各阶段 allocated / reserved / peak 三列数字、
`del` 之后的差值、`gc.collect()` 与 `empty_cache()` 各自的作用。
见 `expected_output.md`。

## 说明

脚本里的 `synchronize` 是**为了对账**，不是测速——它把异步执行拉平，
让"当前阶段"的显存数字可读。不要把它当成性能代码的写法。

## 出处

- 博文：[60-删了张量显存为什么还没降.md](../../../blog/060-显存占用分析：别只盯参数量.md)
- 来源代码块 #5，84 行，md5 `881c4c2957256cc1f71f1ba0fe374c55`

代码与博文逐字一致，未改写。
