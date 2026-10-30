# 第 9 章 · 医疗影像实战

> 篇号 074–081 · 本章 8 篇 · 有独立代码 8 篇 · 全量已实测 3 篇

**本章目标**：三维体积装不进显存时，怎么切、怎么拼、怎么评估。

**必读**：077 → 081

**最常见的误解**：以为评估不用验证。081 里 Dice 与手算逐位一致才算数，空掩膜记 `null` 而不是 1.0。074/078/081 用的是**合成数据**，验证的是几何与公式，不是病例。

## 篇目

| 篇 | 标题 | 代码 | 实测 | 位置 |
|---:|---|---|---|---|
| 074 | CT 看着是一叠图，GPU 要处理的究竟是什么？ | 1 个文件 | 已实测 | [`074-ct-volume-basics/`](074-ct-volume-basics/) |
| 075 | AI 怎样在 CT 里勾出一个器官？ | 2 个文件 | 未实测 | [`075-ct-seg-inference/`](075-ct-seg-inference/) |
| 076 | CT 预处理只有三行，为什么还要写 CUDA？ | 1 个文件 | 部分实测 | [`076-ct-intensity-map/`](076-ct-intensity-map/) |
| 077 | CT 放不进显存，切成小块后怎样拼回去？ | 2 个文件 | 部分实测 | [`077-sliding-window-toy/`](077-sliding-window-toy/) |
| 078 | 器官已经标出来了，AI 还要学什么？ | 1 个文件 | 已实测 | [`078-ct-one-step-train/`](078-ct-one-step-train/) |
| 079 | CT 分割想跑得更快，能少用一些精度吗？ | 2 个文件 | 部分实测 | [`079-ct-amp-check/`](079-ct-amp-check/) |
| 080 | CT 分割已经用上 GPU，为什么还在等数据？ | 1 个文件 | 未实测 | [`080-ct-double-buffer/`](080-ct-double-buffer/) |
| 081 | CUDA 医疗影像实战（8）：CT分割跑完了，怎样判断它做得好不好？ | 1 个文件 | 已实测 | [`081-ct-seg-eval/`](081-ct-seg-eval/) |

## 说明

本目录只放「有独立代码的篇目」（8 / 8 篇）。其余篇目的代码是讲解片段，拼成文件就等于替作者写新代码，因此不单独入库——读 [`blog/`](../../blog/) 里的博文即可，判定标准见[仓库 README 的「剩下 28 篇为什么不收」](../../README.md)。

标「仅博文」的行不是待办，是**判定过的不入库**。

章内依赖顺序与全书路线见 [`docs/learning-path.md`](../../docs/learning-path.md)；逐篇实测证据见 [`docs/reference/verification-status.md`](../../docs/reference/verification-status.md)。
