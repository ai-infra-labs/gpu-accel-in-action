# 配图目录

所有博文共用的图片集中放在 `blog/assets/`，与正文同层管理。
正文里的引用形式是 `](assets/figure-xxx.jpg)`——相对博文文件所在目录。

命名约定：`figure-<主题>-<序号>.<ext>`，前缀对应篇目主题；已有可读名的旧图保持原名。
转换说明：42 张原为 2 MB 级 PNG 截图，已用 `sips` 转 JPEG（quality 85、长边 1600px），
仓库体积从 85 MB 降到 16 MB。PNG 原图未保留——如需重新导出，见各篇 `expected_output.md` 的复现步骤。

## 文件与篇目对应

| 文件 | 篇目 | 图注 |
|---|---|---|
| `figure-fc-forward-01.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-02.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-03.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-04.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-05.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-06.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-07.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-08.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-09.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-10.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-11.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-12.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-13.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-14.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-15.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-16.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-17.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-fc-forward-18.jpg` | 001-PyCUDA全连接层前向与反向传播.md | — |
| `figure-overlap-timeline.png` | 003-CPU与GPU异构计算基础.md | 串行与重叠的执行时间线对比：串行为两段相加，重叠受较长一段（max）决定 |
| `figure-pipeline-batch.png` | 003-CPU与GPU异构计算基础.md | 批处理流水线：H2D、计算、D2H 在时间轴上错开重叠 |
| `figure-conv2d-01.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-02.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-03.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-04.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-05.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-06.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-07.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-08.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-09.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-10.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-11.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-12.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-13.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-14.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-15.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-16.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-17.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-18.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-conv2d-19.jpg` | 007-PyCUDA二维卷积与互相关.md | — |
| `figure-fft-conv-01.jpg` | 016-FFT卷积加速原理与实战.md | — |
| `figure-fft-conv-02.jpg` | 016-FFT卷积加速原理与实战.md | — |
| `figure-fft-conv-03.jpg` | 016-FFT卷积加速原理与实战.md | — |
| `figure-fft-conv-04.jpg` | 016-FFT卷积加速原理与实战.md | — |
| `figure-fft-conv-05.jpg` | 016-FFT卷积加速原理与实战.md | — |
| `figure-dye-progression-cpu-ref.png` | 019-PyCUDA流体格子模拟.md | 染料浓度随时间的演化（CPU 参考实现，与 GPU 采用同一套离散公式与边界规则 |
| `figure-velocity-field-diff-rotate.png` | 019-PyCUDA流体格子模拟.md | 差异旋转速度场：箭头表示方向，颜色表示速率大小（中心十字为旋转中心） |
| `figure-dye-initial-final-cpu.png` | 019-PyCUDA流体格子模拟.md | 墨滴旋转与扩散的初始、最终对比（CPU 参考实现；GPU 采用相同离散公式，输出 |
| `figure-coalesced-access.jpg` | 061-显存带宽与访存模式对性能的影响.md | 相邻线程连续读取时的访存示意 |
| `figure-uncoalesced-access.jpg` | 061-显存带宽与访存模式对性能的影响.md | 地址分散时的非合并访存示意 |
| `figure-matrix-transpose-global.jpg` | 061-显存带宽与访存模式对性能的影响.md | 矩阵转置的全局内存访问示意 |
| `figure-shared-memory-strided.jpg` | 062-共享内存限制与bank冲突.md | 共享内存按行宽对齐时的跨步访问 |
| `figure-bank-structure-32x32.jpg` | 062-共享内存限制与bank冲突.md | 32×32 布局下同一列共享同一 bank |
| `figure-bank-structure-32x33.jpg` | 062-共享内存限制与bank冲突.md | 每行 padding 一列后同一列分散到不同 bank |
| `figure-cuda-streams-overlap.jpg` | 065-打开异步传输为什么还是没变快.md | 两条 stream 上的异步并发执行示意 |
| `figure-cpu-gpu-interconnect.jpg` | 066-两张GPU传数据真的走了NVLink吗.md | CPU、GPU 与 PCIe/NVLink 互联的整机示意 |
| `figure-spacing-display-comparison.png` | 074-CT看着是一叠图：GPU要处理的究竟是什么.md | — |
