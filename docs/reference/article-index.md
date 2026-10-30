# 81 篇 → 仓库目录 映射与进度

> 按章节阅读请看 [学习路线](../learning-path.md)；每篇的依赖与实测状态见 [能力矩阵](capability-matrix.md)。

本表是提取作业清单。状态含义：

| 状态 | 含义 | 处理方式 |
|---|---|---|
| 已入库（已实测） | 代码已入库且跑通，数字记在 expected_output.md | 无需处理 |
| 已入库（已实测·合成数据） | 脚本跑通，但用的是合成输入而非真实病例 | 有真实数据时补一份 |
| 已入库（部分实测） | 多文件篇目，一部分已跑通、另一部分未实测 | 只补未验证的那部分 |
| 已入库（无独立代码） | 只留 README 说明 | 无需处理 |
| 待提取 | 有 ≥60 行完整代码块，可机械拆分 | `scripts/extract_block.py` + pyflakes |
| 待拼装 | 最大代码块 20–59 行，或长块缺 import | 人工合并 v0→v1 |
| 待评估 | 最大代码块 <20 行，疑似无独立代码 | 逐篇判断 |

统计：0 待提取、22 待拼装、6 待评估、53 已入库。

**提取已完成**：52 篇含完整代码块的篇目全部入库（第 2–6 批，共 53 个目录，
含无独立代码的 039）。剩下 22 篇待拼装 + 6 篇待评估 = 28 篇，均只有 <60 行的
讲解片段或完全没有代码，判定为不入库（判据见 README「剩下 28 篇为什么不收」）。

| 篇号 | 标题 | 代码块 | 最大块 | 状态 | 目录 |
|---|---|---:|---:|---|
| 001 | GPU 不认识矩阵：它怎样用 1024 个线程算完一个线性层？ | 29 | 309 | 已入库 | examples/01-gpu-programming/001-fc-forward-backward |
| 002 | 反向传播不是魔法：用 PyCUDA 拆开全连接层的 `dX`、`dW` 与 SGD | 13 | 371 | 已入库 | examples/01-gpu-programming/002-fc-backprop-sgd |
| 003 | GPU 明明很快，程序为什么还在等？一次讲透 CPU–GPU 协同计算 | 9 | 352 | 已入库 | examples/01-gpu-programming/003-cpu-gpu-cooperative |
| 004 | Kernel 已经很快，GPU 为什么还在等？用 PyCUDA 搭建批量数据流水线 | 6 | 221 | 已入库 | examples/01-gpu-programming/004-batch-input-pipeline |
| 005 | 一个线程处理一个像素：用 PyCUDA 手写图像水平翻转 | 12 | 126 | 已入库 | examples/01-gpu-programming/005-image-hflip |
| 006 | 裁剪不是“切一刀”：用 PyCUDA 讲透 ROI 坐标映射 | 13 | 209 | 已入库 | examples/01-gpu-programming/006-image-crop-roi |
| 007 | Kernel 能编译，不代表它在算卷积：拆解 PyCUDA 二维卷积的三个反转 | 26 | 125 | 已入库 | examples/01-gpu-programming/007-conv2d-correlation |
| 008 | PyCUDA 手写 3D 卷积：26 万个输出线程，为什么同一批输入被反复访问？ | 7 | 205 | 已入库 | examples/01-gpu-programming/008-conv3d-pycuda |
| 009 | 两个 Kernel，为什么可以合成一个？ | 11 | 197 | 已入库 | examples/01-gpu-programming/009-fused-preprocess |
| 010 | GPU 不只是“同时算”：为什么线程越多，归约和前缀和反而越难？ | 8 | 435 | 已入库 | examples/01-gpu-programming/010-reduction-prefix-sum |
| 011 | 从局部乘加到共享内存复用：用 PyCUDA 拆开 GPU 矩阵乘法 | 12 | 302 | 已入库（部分实测） | examples/01-gpu-programming/011-tiled-matmul-shared-memory |
| 012 | 一行 `SGEMM`，为什么藏着三套矩阵世界？ | 4 | 42 | 待拼装 | — |
| 013 | GPU 算得快，数据怎么搬也很关键：用 PyCUDA 拆解矩阵乘法的内存流（附完整代码） | 6 | 208 | 已入库 | examples/01-gpu-programming/013-memory-transfer-matmul |
| 014 | 1024 个数，为什么这份朴素 CUDA 实现要启动 55 次 Kernel？ | 12 | 249 | 已入库 | examples/01-gpu-programming/014-bitonic-sort |
| 015 | 把声音拆成频率，究竟发生了什么？ | 9 | 304 | 已入库 | examples/01-gpu-programming/015-naive-dft |
| 016 | FFT 卷积：不再让卷积核一格一格滑，真的会更快吗？ | 12 | 44 | 待拼装 | — |
| 017 | 1024 个 CUDA 线程一起模拟引力，为什么算出的"宇宙"可能是错的？ | 6 | 245 | 已入库 | examples/01-gpu-programming/017-nbody |
| 018 | GPU 算了 1000 次，温度为什么只前进了一步？ | 11 | 233 | 已入库 | examples/01-gpu-programming/018-heat-equation |
| 019 | 26 万个 CUDA 线程，如何推动一滴墨水？ | 18 | 261 | 已入库（部分实测） | examples/01-gpu-programming/019-fluid-grid |
| 020 | 模型压缩全景图：参数少了、FLOPs 降了，为什么模型不一定跑得更快？ | 0 | 0 | 待评估 | — |
| 021 | 模型缩小近 4 倍，推理也会快 4 倍吗？ | 2 | 349 | 已入库 | examples/02-model-compression/021-uint8-weight-quant |
| 022 | Loss 在下降，学生却没学到老师？一篇讲透知识蒸馏 | 11 | 355 | 已入库（部分实测） | examples/02-model-compression/022-cifar10-distillation |
| 023 | 参数缩到原来的 1/106，计算量却只少了 27%：小模型到底小在哪？ | 6 | 92 | 已入库（已实测） | examples/02-model-compression/023-gap-vs-fc-distillation |
| 024 | 权重剪掉一半，模型就能快一倍？用 CUDA 算一笔账 | 17 | 14 | 待评估 | — |
| 025 | 别再搬运那些零：从 CSR 到 CUDA SpMM，拆开稀疏矩阵乘法 | 11 | 27 | 待拼装 | — |
| 026 | 99.99% 都是零：40 GB 的矩阵，如何用 8.4 MB 存下？从 CSR 到 CUDA，讲透稀疏矩阵存储与计算 | 18 | 186 | 已入库 | examples/02-model-compression/026-csr-spmv |
| 027 | 模型训练完了，为什么上线推理还是很慢？ | 1 | 5 | 待评估 | — |
| 028 | PyTorch→ONNX→TensorRT：一条推理链路的验证清单（附完整对照代码） | 5 | 246 | 已入库 | examples/03-inference-serving/028-trt-benchmark |
| 029 | 模型上了 GPU，一次请求的时间都花在哪？一次 TensorRT 对照实验 | 7 | 267 | 已入库 | examples/03-inference-serving/029-trt-pipeline |
| 030 | 三个 AI 模型一起跑，为什么可能比一个一个跑还慢？ | 1 | 37 | 待拼装 | — |
| 031 | 开了 4 条 CUDA 流，GPU 就能同时推理 4 批图片吗？ | 8 | 27 | 待拼装 | — |
| 032 | 10 个推理任务、4 张 GPU：换种排法，整批耗时为何能少 27%？（模拟实验） | 7 | 38 | 待拼装 | — |
| 033 | GPU 明明算得很快，为什么还在等数据？——用 CUDA Streams 搭一条推理流水线 | 6 | 11 | 待评估 | — |
| 034 | 一张 GPU 同时跑三个 AI 模型，为什么请求反而越来越慢？——从 CUDA Streams 到 Triton 推理服务 | 4 | 46 | 待拼装 | — |
| 035 | 4 张 GPU 推理一个模型：复制四份，还是把模型拆开？ | 1 | 27 | 待拼装 | — |
| 036 | 摄像头认出人只要几十毫秒，为什么提醒却晚了两秒？ | 0 | 0 | 待评估 | — |
| 037 | 模型从 40 MB 缩到 10 MB，摄像头为什么还是卡？ | 0 | 0 | 待评估 | — |
| 038 | 把模型里的数字缩小四倍，摄像头就能快四倍吗？ | 1 | 21 | 待拼装 | — |
| 039 | 手机明明有 NPU，为什么 AI 还在 CPU 上跑？ | 1 | 12 | 已入库（无独立代码） | examples/04-edge-inference/039-on-device-npu |
| 040 | 三个模型都很快，摄像头为什么还在看两秒前的画面？ | 1 | 29 | 待拼装 | — |
| 041 | 四个 AI 一起干活，为什么不能快四倍？ | 3 | 50 | 待拼装 | — |
| 042 | 开了四个线程，GPU 就会同时跑吗？ | 9 | 100 | 已入库 | examples/05-parallel-distributed/042-cuda-streams-concurrency |
| 043 | 模型都放进显存了，为什么反而更慢？ | 1 | 94 | 已入库 | examples/05-parallel-distributed/043-vram-contention |
| 044 | 多加一张显卡，任务该交给谁？ | 2 | 65 | 已入库（已实测） | examples/05-parallel-distributed/044-gpu-task-dispatch |
| 045 | 请求一多，AI 为什么开始排长队？ | 1 | 53 | 待拼装 | — |
| 046 | 两张显卡各算一半数据，为什么还要对答案？ | 4 | 58 | 待拼装 | — |
| 047 | GPU 各算各的，结果怎么对齐？讲透 NCCL 通信 | 2 | 66 | 已入库 | examples/05-parallel-distributed/047-nccl-allreduce |
| 048 | 一层矩阵太大，怎样让多张 GPU 合力算？ | 3 | 68 | 已入库（部分实测） | examples/05-parallel-distributed/048-tensor-parallel-linear |
| 049 | 模型分成几段，为什么显卡还会等着没活干？ | 1 | 44 | 待拼装 | — |
| 050 | 模型参数没多大，训练为什么还是爆显存？ | 2 | 57 | 待拼装 | — |
| 051 | 八张显卡，怎样同时分数据、分层、分矩阵？ | 1 | 92 | 已入库（已实测） | examples/05-parallel-distributed/051-parallel-group-check |
| 052 | PyTorch 已经会用 GPU，为什么还要自定义算子？ | 4 | 46 | 待拼装 | — |
| 053 | 一行平方运算，怎样变成 GPU 上的自定义算子？ | 7 | 34 | 待拼装 | — |
| 054 | CUDA 代码跑通了，为什么还没真正接入 PyTorch？ | 13 | 44 | 待拼装 | — |
| 055 | 自定义算子算对了，模型为什么还是学不会？ | 11 | 29 | 待拼装 | — |
| 056 | 手写 CUDA 一定更快吗？三种实现同场比较 | 9 | 62 | 已入库 | examples/06-custom-operators/056-custom-op-vs-eager |
| 057 | 单独运行没问题，为什么一开编译就出错？ | 13 | 34 | 待拼装 | — |
| 058 | 同一段 CUDA，怎样接入 TensorFlow？ | 13 | 39 | 待拼装 | — |
| 059 | 有了 Triton，自定义算子还要写 C++ 吗？ | 16 | 42 | 待拼装 | — |
| 060 | 删了张量，显存为什么还没降？ | 7 | 88 | 已入库 | examples/07-memory-and-transfer/060-tensor-memory-ledger |
| 061 | 同样搬一张表，GPU 为什么还要看访问方式？ | 4 | 114 | 已入库 | examples/07-memory-and-transfer/061-coalescing-lab |
| 062 | 共享内存这么快，为什么不能全放进去？ | 4 | 68 | 已入库 | examples/07-memory-and-transfer/062-shared-memory-limit |
| 063 | 一组线程读同一个数，GPU 有什么捷径？ | 3 | 103 | 已入库 | examples/07-memory-and-transfer/063-warp-broadcast |
| 064 | 数据还没送到，GPU 已经等急了 | 4 | 92 | 已入库 | examples/07-memory-and-transfer/064-host-to-device-wait |
| 065 | 打开异步传输，为什么还是没变快？ | 3 | 127 | 已入库 | examples/07-memory-and-transfer/065-async-transfer-overlap |
| 066 | 两张 GPU 传数据，真的走了 NVLink 吗？ | 3 | 101 | 已入库 | examples/07-memory-and-transfer/066-nvlink-copy |
| 067 | 模型导出成功，为什么结果却变了？ | 1 | 126 | 已入库（已实测） | examples/08-tensorrt-deploy/067-onnx-export-alignment |
| 068 | TensorRT 没改模型，为什么还能更快？ | 1 | 69 | 已入库 | examples/08-tensorrt-deploy/068-engine-build |
| 069 | 引擎已经生成，为什么推理还会报错？ | 3 | 129 | 已入库 | examples/08-tensorrt-deploy/069-engine-runtime |
| 070 | 推理只要 1 毫秒，应用为什么还在等？ | 2 | 123 | 已入库 | examples/08-tensorrt-deploy/070-app-latency |
| 071 | 模型变成 INT8，精度到底丢在哪里？ | 3 | 64 | 已入库 | examples/08-tensorrt-deploy/071-int8-quant |
| 072 | 给权重做剪枝，CUDA 到底加速了哪一步？ | 5 | 115 | 已入库 | examples/08-tensorrt-deploy/072-pruning-kernel |
| 073 | 权重删了一半，模型为什么没快？ | 3 | 93 | 已入库（已实测） | examples/08-tensorrt-deploy/073-pruning-no-speedup |
| 074 | CT 看着是一叠图，GPU 要处理的究竟是什么？ | 3 | 104 | 已入库（已实测·合成数据） | examples/09-medical-imaging/074-ct-volume-basics |
| 075 | AI 怎样在 CT 里勾出一个器官？ | 5 | 76 | 已入库 | examples/09-medical-imaging/075-ct-seg-inference |
| 076 | CT 预处理只有三行，为什么还要写 CUDA？ | 2 | 163 | 已入库（部分实测） | examples/09-medical-imaging/076-ct-intensity-map |
| 077 | CT 放不进显存，切成小块后怎样拼回去？ | 2 | 125 | 已入库（部分实测） | examples/09-medical-imaging/077-sliding-window-toy |
| 078 | 器官已经标出来了，AI 还要学什么？ | 2 | 126 | 已入库（已实测·合成数据） | examples/09-medical-imaging/078-ct-one-step-train |
| 079 | CT 分割想跑得更快，能少用一些精度吗？ | 4 | 173 | 已入库（部分实测） | examples/09-medical-imaging/079-ct-amp-check |
| 080 | CT 分割已经用上 GPU，为什么还在等数据？ | 3 | 141 | 已入库 | examples/09-medical-imaging/080-ct-double-buffer |
| 081 | CUDA 医疗影像实战（8）：CT分割跑完了，怎样判断它做得好不好？ | 1 | 151 | 已入库（已实测·合成数据） | examples/09-medical-imaging/081-ct-seg-eval |
