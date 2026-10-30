# 080 · CT 分割已经用上 GPU，为什么还在等数据

GPU 在算的时候，H2D 拷贝是闲着的；拷贝的时候，GPU 是闲着的。如果两者在同一条流
上串行，整批耗时就是"拷贝之和 + 计算之和"。双缓冲要做的事只有一件：
**让第 n 批的计算和第 n+1 批的拷贝同时发生**。

本篇用一个 2 层 Conv3d 的假模型和合成窗口，把两种编排放在一起比：

| 编排 | 结构 |
|---|---|
| 串行（同流） | `H2D → 计算 → H2D → 计算 …`，全部在默认流上 |
| 双缓冲 | 两个设备槽位 + 两条流，拷贝与计算交替重叠 |

## 前置

078、077（推理与窗口）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `double_buffer_infer.py` | 锁页主机批次 + 双槽双流编排 + 正确性对照 + median/p10/p90 | 博文代码块 #4 |

## 运行

```bash
python -m pip install -r ../../requirements/advanced.txt
python double_buffer_infer.py --roi 32 --batches 12 --repeats 5 --out copy_compute_report.json
python double_buffer_infer.py --trace      # 额外导出 Chrome trace
```

参数：`--roi`（默认 32）、`--batches`（默认 12）、`--repeats`（默认 5）、
`--trace`、`--out`（默认 `copy_compute_report.json`）。
每批形状是 `(2, 1, roi, roi, roi)`——两个合成窗口。

## 环境要求

- `torch` + **CUDA**（`requirements/advanced.txt`）。脚本在无 CUDA 时直接抛
  `RuntimeError`，提示改看 CPU 调度例子。
- 需要锁页内存（`pin_memory()`），这是双缓冲能重叠的前提

## 观察点

- 双缓冲 vs 串行的比值，上限由「拷贝时间 / 计算时间」决定。
- p10 与 p90 的宽度反映抖动；只报中位数会掩盖长尾。

## 验证状态

**未实测。** 本机无 CUDA，脚本在启动阶段即拒绝运行（已实测该守卫会触发）。

需要记录：两种编排的 median/p10/p90、正确性对照结果、trace 里的重叠情况。
见 `expected_output.md`。

## 出处

- 博文：[80-CT分割已经用上GPU为什么还在等数据.md](../../../blog/080-CT分割已经用上GPU为什么还在等数据.md)
- 来源代码块 #4，141 行，md5 `fb1966a1b135b53d6a670e9928590392`

代码逐字一致。**博文未指定文件名**，`double_buffer_infer.py` 为仓库新增命名。
