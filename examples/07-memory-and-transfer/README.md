# 第 7 章 · 访存与传输

> 篇号 060–066 · 本章 7 篇 · 有独立代码 7 篇 · 全量已实测 0 篇

**本章目标**：六个可独立验证的机制：显存记账、合并访存、共享内存容量、常量内存广播、页锁定内存、异步重叠。

**必读**：061 → 062 → 063 → 065

**最常见的误解**：以为「加了异步」就一定重叠。065 的判据是总时间接近 `max(拷贝, 计算)`，不是代码里出现了 stream。

## 篇目

| 篇 | 标题 | 代码 | 实测 | 位置 |
|---:|---|---|---|---|
| 060 | 删了张量，显存为什么还没降？ | 1 个文件 | 未实测 | [`060-tensor-memory-ledger/`](060-tensor-memory-ledger/) |
| 061 | 同样搬一张表，GPU 为什么还要看访问方式？ | 1 个文件 | 未实测 | [`061-coalescing-lab/`](061-coalescing-lab/) |
| 062 | 共享内存这么快，为什么不能全放进去？ | 1 个文件 | 未实测 | [`062-shared-memory-limit/`](062-shared-memory-limit/) |
| 063 | 一组线程读同一个数，GPU 有什么捷径？ | 1 个文件 | 未实测 | [`063-warp-broadcast/`](063-warp-broadcast/) |
| 064 | 数据还没送到，GPU 已经等急了 | 1 个文件 | 未实测 | [`064-host-to-device-wait/`](064-host-to-device-wait/) |
| 065 | 打开异步传输，为什么还是没变快？ | 1 个文件 | 未实测 | [`065-async-transfer-overlap/`](065-async-transfer-overlap/) |
| 066 | 两张 GPU 传数据，真的走了 NVLink 吗？ | 1 个文件 | 未实测 | [`066-nvlink-copy/`](066-nvlink-copy/) |

## 说明

本目录只放「有独立代码的篇目」（7 / 7 篇）。其余篇目的代码是讲解片段，拼成文件就等于替作者写新代码，因此不单独入库——读 [`blog/`](../../blog/) 里的博文即可，判定标准见[仓库 README 的「剩下 28 篇为什么不收」](../../README.md)。

标「仅博文」的行不是待办，是**判定过的不入库**。

章内依赖顺序与全书路线见 [`docs/learning-path.md`](../../docs/learning-path.md)；逐篇实测证据见 [`docs/reference/verification-status.md`](../../docs/reference/verification-status.md)。
