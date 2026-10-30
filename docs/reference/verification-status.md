# 实测覆盖看板（自动生成，勿手改）

由 `python3 scripts/gen_verify.py` 从各篇 `expected_output.md` 的 `**状态：…**`
汇总而成。**状态只认文件里的声明**：代码跑通了但没写进 `expected_output.md`，
表里仍是「未实测」——证据不落盘就不算数。

## 覆盖率

- 已落地篇目目录：**53** 篇（全书 81 篇，其余 28 篇为片段型或纯文字篇）
- 全量已实测：**8** 篇（**15%**）
- 部分已实测：**7** 篇（一份代码里只有主机端那半跑过）
- 未实测：**37** 篇
- 不适用（目录里无代码）：**1** 篇
- 未声明状态：**0** 篇

**两个口径要分开看**：只算全量是通过率 **15%**；把「部分实测」也算上则到 **28%**。对外只应引用前者——分母固定，所以它只会因为真跑通一篇并写入证据而上升，
不会因为新增概念篇而下降。

## 分章

| 章 | 章名 | 篇号 | 有目录 | 全量实测 | 部分实测 | 全量通过率 |
|---:|---|---|---:|---:|---:|---:|
| 1 | GPU 编程基础 | 001–019 | 17 | 0 | 2 | 0% |
| 2 | 模型压缩 | 020–026 | 4 | 1 | 1 | 25% |
| 3 | 推理服务与部署 | 027–035 | 2 | 0 | 0 | 0% |
| 4 | 端侧推理 | 036–041 | 1 | 0 | 0 | 0% |
| 5 | 并行与分布式 | 042–051 | 6 | 2 | 1 | 33% |
| 6 | 自定义算子 | 052–059 | 1 | 0 | 0 | 0% |
| 7 | 访存与传输 | 060–066 | 7 | 0 | 0 | 0% |
| 8 | TensorRT 部署 | 067–073 | 7 | 2 | 0 | 29% |
| 9 | 医疗影像实战 | 074–081 | 8 | 3 | 3 | 38% |

## 逐篇明细

| 篇 | 标题 | 章 | 状态 | 声明原文 | 证据 |
|---:|---|---:|---|---|---|
| 001 | GPU 不认识矩阵：它怎样用 1024 个线程… | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/001-fc-forward-backward/expected_output.md` |
| 002 | 反向传播不是魔法：用 PyCUDA 拆开全连接… | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/002-fc-backprop-sgd/expected_output.md` |
| 003 | GPU 明明很快，程序为什么还在等？一次讲透 … | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/003-cpu-gpu-cooperative/expected_output.md` |
| 004 | Kernel 已经很快，GPU 为什么还在等？… | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/004-batch-input-pipeline/expected_output.md` |
| 005 | 一个线程处理一个像素：用 PyCUDA 手写图… | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/005-image-hflip/expected_output.md` |
| 006 | 裁剪不是“切一刀”：用 PyCUDA 讲透 R… | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/006-image-crop-roi/expected_output.md` |
| 007 | Kernel 能编译，不代表它在算卷积：拆解 … | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/007-conv2d-correlation/expected_output.md` |
| 008 | PyCUDA 手写 3D 卷积：26 万个输出… | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/008-conv3d-pycuda/expected_output.md` |
| 009 | 两个 Kernel，为什么可以合成一个？ | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/009-fused-preprocess/expected_output.md` |
| 010 | GPU 不只是“同时算”：为什么线程越多，归约… | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/010-reduction-prefix-sum/expected_output.md` |
| 011 | 从局部乘加到共享内存复用：用 PyCUDA 拆… | 1 | 部分实测 | 部分实测（主机端索引校验已实测，GPU Kernel 未实… | `examples/01-gpu-programming/011-tiled-matmul-shared-memory/expected_output.md` |
| 013 | GPU 算得快，数据怎么搬也很关键：用 PyC… | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/013-memory-transfer-matmul/expected_output.md` |
| 014 | 1024 个数，为什么这份朴素 CUDA 实现… | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/014-bitonic-sort/expected_output.md` |
| 015 | 把声音拆成频率，究竟发生了什么？ | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/015-naive-dft/expected_output.md` |
| 017 | 1024 个 CUDA 线程一起模拟引力，为什… | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/017-nbody/expected_output.md` |
| 018 | GPU 算了 1000 次，温度为什么只前进了… | 1 | 未实测 | 未实测 | `examples/01-gpu-programming/018-heat-equation/expected_output.md` |
| 019 | 26 万个 CUDA 线程，如何推动一滴墨水？ | 1 | 部分实测 | 部分实测（CPU 参考实现已实测，GPU Kernel 未… | `examples/01-gpu-programming/019-fluid-grid/expected_output.md` |
| 021 | 模型缩小近 4 倍，推理也会快 4 倍吗？ | 2 | 未实测 | 未实测 | `examples/02-model-compression/021-uint8-weight-quant/expected_output.md` |
| 022 | Loss 在下降，学生却没学到老师？一篇讲透知… | 2 | 部分实测 | 自检已实测；完整对照未实测 | `examples/02-model-compression/022-cifar10-distillation/expected_output.md` |
| 023 | 参数缩到原来的 1/106，计算量却只少了 2… | 2 | 已实测 | 已实测通过 | `examples/02-model-compression/023-gap-vs-fc-distillation/expected_output.md` |
| 026 | 99.99% 都是零：40 GB 的矩阵，如何… | 2 | 未实测 | 未实测 | `examples/02-model-compression/026-csr-spmv/expected_output.md` |
| 028 | PyTorch→ONNX→TensorRT：一… | 3 | 未实测 | 未实测（需 TensorRT 10.x + NVIDIA … | `examples/03-inference-serving/028-trt-benchmark/expected_output.md` |
| 029 | 模型上了 GPU，一次请求的时间都花在哪？一次… | 3 | 未实测 | 未实测（需 TensorRT 8.x + NVIDIA G… | `examples/03-inference-serving/029-trt-pipeline/expected_output.md` |
| 039 | 手机明明有 NPU，为什么 AI 还在 CPU… | 4 | 不适用 | — | `examples/04-edge-inference/039-on-device-npu/expected_output.md` |
| 042 | 开了四个线程，GPU 就会同时跑吗？ | 5 | 未实测 | 未实测 | `examples/05-parallel-distributed/042-cuda-streams-concurrency/expected_output.md` |
| 043 | 模型都放进显存了，为什么反而更慢？ | 5 | 未实测 | 未实测 | `examples/05-parallel-distributed/043-vram-contention/expected_output.md` |
| 044 | 多加一张显卡，任务该交给谁？ | 5 | 已实测 | 已实测通过 | `examples/05-parallel-distributed/044-gpu-task-dispatch/expected_output.md` |
| 047 | GPU 各算各的，结果怎么对齐？讲透 NCCL… | 5 | 未实测 | 未实测（需 2 张 NVIDIA GPU + torchr… | `examples/05-parallel-distributed/047-nccl-allreduce/expected_output.md` |
| 048 | 一层矩阵太大，怎样让多张 GPU 合力算？ | 5 | 部分实测 | tp_split_check.py 已实测；tp_line… | `examples/05-parallel-distributed/048-tensor-parallel-linear/expected_output.md` |
| 051 | 八张显卡，怎样同时分数据、分层、分矩阵？ | 5 | 已实测 | 已实测通过 | `examples/05-parallel-distributed/051-parallel-group-check/expected_output.md` |
| 056 | 手写 CUDA 一定更快吗？三种实现同场比较 | 6 | 未实测 | 未实测（需 NVIDIA GPU + nvcc + tor… | `examples/06-custom-operators/056-custom-op-vs-eager/expected_output.md` |
| 060 | 删了张量，显存为什么还没降？ | 7 | 未实测 | 未实测 | `examples/07-memory-and-transfer/060-tensor-memory-ledger/expected_output.md` |
| 061 | 同样搬一张表，GPU 为什么还要看访问方式？ | 7 | 未实测 | 未实测 | `examples/07-memory-and-transfer/061-coalescing-lab/expected_output.md` |
| 062 | 共享内存这么快，为什么不能全放进去？ | 7 | 未实测 | 未实测（需 numba-cuda） | `examples/07-memory-and-transfer/062-shared-memory-limit/expected_output.md` |
| 063 | 一组线程读同一个数，GPU 有什么捷径？ | 7 | 未实测 | 未实测（需 numba-cuda） | `examples/07-memory-and-transfer/063-warp-broadcast/expected_output.md` |
| 064 | 数据还没送到，GPU 已经等急了 | 7 | 未实测 | 未实测 | `examples/07-memory-and-transfer/064-host-to-device-wait/expected_output.md` |
| 065 | 打开异步传输，为什么还是没变快？ | 7 | 未实测 | 未实测 | `examples/07-memory-and-transfer/065-async-transfer-overlap/expected_output.md` |
| 066 | 两张 GPU 传数据，真的走了 NVLink … | 7 | 未实测 | 未实测（需 2 张 NVIDIA GPU） | `examples/07-memory-and-transfer/066-nvlink-copy/expected_output.md` |
| 067 | 模型导出成功，为什么结果却变了？ | 8 | 已实测 | 已实测通过 | `examples/08-tensorrt-deploy/067-onnx-export-alignment/expected_output.md` |
| 068 | TensorRT 没改模型，为什么还能更快？ | 8 | 未实测 | 未实测（需 TensorRT 11.x + NVIDIA … | `examples/08-tensorrt-deploy/068-engine-build/expected_output.md` |
| 069 | 引擎已经生成，为什么推理还会报错？ | 8 | 未实测 | 未实测（需 TensorRT 11.x + 已构建的 en… | `examples/08-tensorrt-deploy/069-engine-runtime/expected_output.md` |
| 070 | 推理只要 1 毫秒，应用为什么还在等？ | 8 | 未实测 | 未实测（需 TensorRT 11.x + 已构建的 en… | `examples/08-tensorrt-deploy/070-app-latency/expected_output.md` |
| 071 | 模型变成 INT8，精度到底丢在哪里？ | 8 | 未实测 | 未实测（需 nvidia-modelopt） | `examples/08-tensorrt-deploy/071-int8-quant/expected_output.md` |
| 072 | 给权重做剪枝，CUDA 到底加速了哪一步？ | 8 | 未实测 | 未实测 | `examples/08-tensorrt-deploy/072-pruning-kernel/expected_output.md` |
| 073 | 权重删了一半，模型为什么没快？ | 8 | 已实测 | 已实测通过 | `examples/08-tensorrt-deploy/073-pruning-no-speedup/expected_output.md` |
| 074 | CT 看着是一叠图，GPU 要处理的究竟是什么？ | 9 | 已实测 | 已实测通过 | `examples/09-medical-imaging/074-ct-volume-basics/expected_output.md` |
| 075 | AI 怎样在 CT 里勾出一个器官？ | 9 | 未实测 | 未实测（需数据 + MONAI 权重） | `examples/09-medical-imaging/075-ct-seg-inference/expected_output.md` |
| 076 | CT 预处理只有三行，为什么还要写 CUDA？ | 9 | 部分实测 | CPU 检查已实测通过；CuPy / CUDA 部分未实测 | `examples/09-medical-imaging/076-ct-intensity-map/expected_output.md` |
| 077 | CT 放不进显存，切成小块后怎样拼回去？ | 9 | 部分实测 | 部分实测（NumPy 教学版已实测，MONAI 连接检查未… | `examples/09-medical-imaging/077-sliding-window-toy/expected_output.md` |
| 078 | 器官已经标出来了，AI 还要学什么？ | 9 | 已实测 | 已实测通过 | `examples/09-medical-imaging/078-ct-one-step-train/expected_output.md` |
| 079 | CT 分割想跑得更快，能少用一些精度吗？ | 9 | 部分实测 | 格式检查已实测通过；AMP 部分未实测 | `examples/09-medical-imaging/079-ct-amp-check/expected_output.md` |
| 080 | CT 分割已经用上 GPU，为什么还在等数据？ | 9 | 未实测 | 未实测 | `examples/09-medical-imaging/080-ct-double-buffer/expected_output.md` |
| 081 | CUDA 医疗影像实战（8）：CT分割跑完了，… | 9 | 已实测 | 已实测通过 | `examples/09-medical-imaging/081-ct-seg-eval/expected_output.md` |

## 未入库的篇目

以下 28 篇没有独立可运行代码（判定理由见 README「剩下 28 篇为什么不收」）：

012、016、020、024、025、027、030、031、032、033、034、035、036、037、038、040、041、045、046、049、050、052、053、054、055、057、058、059

## 怎么把一篇从「未实测」推到「已实测」

1. 在 N 卡机器上按该篇 `README.md` 的「运行」小节执行；
2. 把真实输出填进 `expected_output.md`，首行写 `**状态：已实测（硬件，日期）**`；
3. 同时写清「这份证据不能说明什么」（例如合成数据、单卡、未覆盖 OOM 路径）；
4. `make check` —— 本表会在 `--check` 阶段随之更新，忘记重跑会直接失败。

只改 `expected_output.md` 不改代码，是这个仓库唯一允许的「升级」方式：
代码必须与博文逐字一致（见 `docs/maintaining/provenance.json` 与 `scripts/build_from_blocks.py`）。
