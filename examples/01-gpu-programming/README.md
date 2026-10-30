# 第 1 章 · GPU 编程基础

> 篇号 001–019 · 本章 19 篇 · 有独立代码 17 篇 · 全量已实测 0 篇

**本章目标**：GPU 里没有矩阵，只有线程编号和一段连续显存。这一章把「我怎么知道第 i 个线程该算哪个数」变成不需要想的事。

**必读**：001 → 010 → 017

**最常见的误解**：以为线程越多越快。006 的 `grid=(7,7)` 会多启动 144 个线程靠边界判断挡掉——多出来的线程不是免费的。

## 篇目

| 篇 | 标题 | 代码 | 实测 | 位置 |
|---:|---|---|---|---|
| 001 | GPU 不认识矩阵：它怎样用 1024 个线程算完一个线性层？ | 1 个文件 | 未实测 | [`001-fc-forward-backward/`](001-fc-forward-backward/) |
| 002 | 反向传播不是魔法：用 PyCUDA 拆开全连接层的 `dX`、`dW` 与 SGD | 1 个文件 | 未实测 | [`002-fc-backprop-sgd/`](002-fc-backprop-sgd/) |
| 003 | GPU 明明很快，程序为什么还在等？一次讲透 CPU–GPU 协同计算 | 1 个文件 | 未实测 | [`003-cpu-gpu-cooperative/`](003-cpu-gpu-cooperative/) |
| 004 | Kernel 已经很快，GPU 为什么还在等？用 PyCUDA 搭建批量数据流水线 | 1 个文件 | 未实测 | [`004-batch-input-pipeline/`](004-batch-input-pipeline/) |
| 005 | 一个线程处理一个像素：用 PyCUDA 手写图像水平翻转 | 1 个文件 | 未实测 | [`005-image-hflip/`](005-image-hflip/) |
| 006 | 裁剪不是“切一刀”：用 PyCUDA 讲透 ROI 坐标映射 | 1 个文件 | 未实测 | [`006-image-crop-roi/`](006-image-crop-roi/) |
| 007 | Kernel 能编译，不代表它在算卷积：拆解 PyCUDA 二维卷积的三个反转 | 1 个文件 | 未实测 | [`007-conv2d-correlation/`](007-conv2d-correlation/) |
| 008 | PyCUDA 手写 3D 卷积：26 万个输出线程，为什么同一批输入被反复访问？ | 1 个文件 | 未实测 | [`008-conv3d-pycuda/`](008-conv3d-pycuda/) |
| 009 | 两个 Kernel，为什么可以合成一个？ | 1 个文件 | 未实测 | [`009-fused-preprocess/`](009-fused-preprocess/) |
| 010 | GPU 不只是“同时算”：为什么线程越多，归约和前缀和反而越难？ | 2 个文件 | 未实测 | [`010-reduction-prefix-sum/`](010-reduction-prefix-sum/) |
| 011 | 从局部乘加到共享内存复用：用 PyCUDA 拆开 GPU 矩阵乘法 | 2 个文件 | 部分实测 | [`011-tiled-matmul-shared-memory/`](011-tiled-matmul-shared-memory/) |
| 012 | 一行 `SGEMM`，为什么藏着三套矩阵世界？ | 无 | — | 仅博文 |
| 013 | GPU 算得快，数据怎么搬也很关键：用 PyCUDA 拆解矩阵乘法的内存流（附完整代码） | 1 个文件 | 未实测 | [`013-memory-transfer-matmul/`](013-memory-transfer-matmul/) |
| 014 | 1024 个数，为什么这份朴素 CUDA 实现要启动 55 次 Kernel？ | 1 个文件 | 未实测 | [`014-bitonic-sort/`](014-bitonic-sort/) |
| 015 | 把声音拆成频率，究竟发生了什么？ | 1 个文件 | 未实测 | [`015-naive-dft/`](015-naive-dft/) |
| 016 | FFT 卷积：不再让卷积核一格一格滑，真的会更快吗？ | 无 | — | 仅博文 |
| 017 | 1024 个 CUDA 线程一起模拟引力，为什么算出的"宇宙"可能是错的？ | 1 个文件 | 未实测 | [`017-nbody/`](017-nbody/) |
| 018 | GPU 算了 1000 次，温度为什么只前进了一步？ | 1 个文件 | 未实测 | [`018-heat-equation/`](018-heat-equation/) |
| 019 | 26 万个 CUDA 线程，如何推动一滴墨水？ | 2 个文件 | 部分实测 | [`019-fluid-grid/`](019-fluid-grid/) |

## 说明

本目录只放「有独立代码的篇目」（17 / 19 篇）。其余篇目的代码是讲解片段，拼成文件就等于替作者写新代码，因此不单独入库——读 [`blog/`](../../blog/) 里的博文即可，判定标准见[仓库 README 的「剩下 28 篇为什么不收」](../../README.md)。

标「仅博文」的行不是待办，是**判定过的不入库**。

章内依赖顺序与全书路线见 [`docs/learning-path.md`](../../docs/learning-path.md)；逐篇实测证据见 [`docs/reference/verification-status.md`](../../docs/reference/verification-status.md)。
