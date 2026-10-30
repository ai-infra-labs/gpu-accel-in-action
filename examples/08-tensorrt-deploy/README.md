# 第 8 章 · TensorRT 部署

> 篇号 067–073 · 本章 7 篇 · 有独立代码 7 篇 · 全量已实测 2 篇

**本章目标**：一条能核对结果的推理链路：导出 → 构建 → 运行 → 计时 → 量化。

**必读**：067 → 069 → 070 → 073

**最常见的误解**：以为「导出成功」等于「数值一致」。067 要用 13 组用例才敢说后者。

## 篇目

| 篇 | 标题 | 代码 | 实测 | 位置 |
|---:|---|---|---|---|
| 067 | 模型导出成功，为什么结果却变了？ | 1 个文件 | 已实测 | [`067-onnx-export-alignment/`](067-onnx-export-alignment/) |
| 068 | TensorRT 没改模型，为什么还能更快？ | 1 个文件 | 未实测 | [`068-engine-build/`](068-engine-build/) |
| 069 | 引擎已经生成，为什么推理还会报错？ | 1 个文件 | 未实测 | [`069-engine-runtime/`](069-engine-runtime/) |
| 070 | 推理只要 1 毫秒，应用为什么还在等？ | 1 个文件 | 未实测 | [`070-app-latency/`](070-app-latency/) |
| 071 | 模型变成 INT8，精度到底丢在哪里？ | 1 个文件 | 未实测 | [`071-int8-quant/`](071-int8-quant/) |
| 072 | 给权重做剪枝，CUDA 到底加速了哪一步？ | 2 个文件 | 未实测 | [`072-pruning-kernel/`](072-pruning-kernel/) |
| 073 | 权重删了一半，模型为什么没快？ | 1 个文件 | 已实测 | [`073-pruning-no-speedup/`](073-pruning-no-speedup/) |

## 说明

本目录只放「有独立代码的篇目」（7 / 7 篇）。其余篇目的代码是讲解片段，拼成文件就等于替作者写新代码，因此不单独入库——读 [`blog/`](../../blog/) 里的博文即可，判定标准见[仓库 README 的「剩下 28 篇为什么不收」](../../README.md)。

标「仅博文」的行不是待办，是**判定过的不入库**。

章内依赖顺序与全书路线见 [`docs/learning-path.md`](../../docs/learning-path.md)；逐篇实测证据见 [`docs/reference/verification-status.md`](../../docs/reference/verification-status.md)。
