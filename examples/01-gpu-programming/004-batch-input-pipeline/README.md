# 004 · Kernel 已经很快，GPU 为什么还在等？

Kernel 自身只要几十微秒，端到端却慢一个数量级——时间花在数据准备、传输与
排队上。单看 Kernel 时间会得出"GPU 很快"的错误结论。

本篇搭一条批量输入流水线：主机侧准备批次、H2D 传输、Kernel 执行、稳态吞吐统计。

## 前置

003（重叠与流水线的前提）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `batch_input_pipeline.py` | 裁剪 + 归一化 kernel + 端到端流水线 | 博文代码块 #10 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python batch_input_pipeline.py
```

打印模拟端到端耗时与稳态吞吐量（images/s），无命令行参数。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit（`SourceModule` 编译依赖 `nvcc`）
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- 把「Kernel 时间 / 端到端时间」算出来：这个比值越小，优化 Kernel 的收益越有限。
- 第一批若包含 context 初始化与 JIT 编译，吞吐会被低估——那部分要单独剔出来。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查。

需要记录：端到端耗时、稳态吞吐、Kernel 占比。见 `expected_output.md`。

## 出处

- 博文：[04-pycuda-batch-input-pipeline-light-tech.md](../../../blog/004-PyCUDA批处理输入流水线.md)
- 来源代码块 #10，217 行，md5 `fcfa892b5f68218a2f4a66e90fdc3645`

代码逐字一致。**博文未指定文件名**，`batch_input_pipeline.py` 为仓库新增命名。
