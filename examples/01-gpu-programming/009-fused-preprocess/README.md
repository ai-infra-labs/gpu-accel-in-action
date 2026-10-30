# 009 · 两个 Kernel，为什么可以合成一个？

裁剪、归一化这类逐像素变换，如果拆成两个 Kernel，每个中间结果都要写回显存
再读出来——多一次往返就是多一倍带宽。融合（fusion）的收益不是"少写几行代码"，
而是**少一次显存往返**。

本篇把 batch + center crop + normalize 三步合成一个 Kernel，并与非融合路径
对比耗时与端到端时间。

## 前置

007、008（知道拆成两个 Kernel 的代价）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `fused_preprocess.py` | 融合预处理 kernel + 参数校验 + 计时 | 博文代码块 #30 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python fused_preprocess.py
```

打印 Block/Grid、Kernel 时间、端到端时间。脚本在模块级执行，无 `main()`。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- 融合版与非融合版的端到端差，应约等于一次中间结果的显存往返。
- 只看 Kernel 时间会看不出收益：差值主要在中间的写回/读出。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查。

需要记录：Kernel 时间、端到端时间、以及（若有）融合前后的对比。
见 `expected_output.md`。

## 出处

- 博文：[09-cuda-fused-batch-center-crop-normalize-light-tech.md](../../../blog/009-CUDA批量中心裁剪与归一化融合实战.md)
- 来源代码块 #30，193 行，md5 `f8115c4bf2fc31a5d45f0bbcaddde335`

代码逐字一致。**博文未指定文件名**，`fused_preprocess.py` 为仓库新增命名。
