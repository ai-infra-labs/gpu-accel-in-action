# 能力矩阵（自动生成，勿手改）

由 `python3 scripts/gen_atlas.py` 从目录、`README.md`、`expected_output.md` 与
`scripts/static_check.py` 汇总而成。改完代码或文档重跑一次即可；
`python3 scripts/gen_atlas.py --check` 会在表过期时报错。

共 53 个篇目。读表三条约定：

- **依赖层** CPU / GPU / 特殊 对应 `requirements/` 的三个文件；`CPU*` 表示脚本依赖在
  CPU 层，但调用了 CUDA API——装完依赖仍需显卡。
- **硬件** 写“CPU 可跑”不代表已实测，只代表不强制要卡。
- **验证状态** 取自各篇 `expected_output.md` 的首行声明，是唯一算数的列。

| 章 | 篇 | 目录 | 代码 | 依赖层 | 硬件 | 验证状态 | 机制 | 前置 |
|---:|---:|---|---|---|---|---|---|---|
| 1 | 001 | `001-fc-forward-backward` | 1 个 / 309 行 | GPU | 需 GPU | 未实测 | thread-mapping | 无。会写 NumPy 矩阵乘就能读；本篇是后面所有 Kernel 的骨架。 |
| 1 | 002 | `002-fc-backprop-sgd` | 1 个 / 371 行 | GPU | 需 GPU | 未实测 | thread-mapping、coalescing | 001（同样的 y=Xw，本篇只做反向与更新）。 |
| 1 | 003 | `003-cpu-gpu-cooperative` | 1 个 / 352 行 | 特殊 | 需 GPU | 未实测 | async-copy、latency-breakdown | 001、002（先知道单个 Kernel 多快，才看得出「等」在哪）。 |
| 1 | 004 | `004-batch-input-pipeline` | 1 个 / 221 行 | GPU | 需 GPU | 未实测 | latency-breakdown、transfer | 003（重叠与流水线的前提）。 |
| 1 | 005 | `005-image-hflip` | 1 个 / 126 行 | GPU | 需 GPU | 未实测 | thread-mapping | 001（线程与索引的映射方式）。 |
| 1 | 006 | `006-image-crop-roi` | 1 个 / 209 行 | GPU | 需 GPU | 未实测 | thread-mapping | 005（同样是二维线程映射）。 |
| 1 | 007 | `007-conv2d-correlation` | 1 个 / 125 行 | GPU | 需 GPU | 未实测 | thread-mapping | 006（二维索引与边界）。 |
| 1 | 008 | `008-conv3d-pycuda` | 1 个 / 205 行 | GPU | 需 GPU | 未实测 | thread-mapping、memory-reuse | 007（先建立二维卷积的直觉）。 |
| 1 | 009 | `009-fused-preprocess` | 1 个 / 197 行 | GPU | 需 GPU | 未实测 | fusion、memory-reuse | 007、008（知道拆成两个 Kernel 的代价）。 |
| 1 | 010 | `010-reduction-prefix-sum` | 2 个 / 720 行 | GPU | 需 GPU | 未实测 | reduction、sync | 001（线程块与网格）。 |
| 1 | 011 | `011-tiled-matmul-shared-memory` | 2 个 / 400 行 | GPU | 需 GPU | 部分实测（主机端索引校验已实测，GPU Kernel 未实测，逐脚… | shared-memory、sync | 010（共享内存与 __syncthreads）。 |
| 1 | 013 | `013-memory-transfer-matmul` | 1 个 / 208 行 | GPU | 需 GPU | 未实测 | transfer、latency-breakdown | 003（传输与计算的时间拆分）。 |
| 1 | 014 | `014-bitonic-sort` | 1 个 / 249 行 | GPU | 需 GPU | 未实测 | launch-overhead、sync | 010（块内归约与同步）。 |
| 1 | 015 | `015-naive-dft` | 1 个 / 304 行 | GPU | 需 GPU | 未实测 | shared-memory | 010（共享内存搬运与同步）。 |
| 1 | 017 | `017-nbody` | 1 个 / 245 行 | GPU | 需 GPU | 未实测 | data-race、thread-mapping | 001、010（线程映射与归约）。 |
| 1 | 018 | `018-heat-equation` | 1 个 / 233 行 | GPU | 需 GPU | 未实测 | sync、stability | 017（双 Kernel 串接与数值稳定性）。 |
| 1 | 019 | `019-fluid-grid` | 2 个 / 336 行 | GPU | 需 GPU | 部分实测（CPU 参考实现已实测，GPU Kernel 未实测，逐… | sync、boundary | 018（显式差分与双缓冲）。 |
| 2 | 021 | `021-uint8-weight-quant` | 1 个 / 348 行 | 特殊 | 需 GPU | 未实测 | weight-quant、dsl-numba | 020（模型压缩的全景，知道量化在整张图里的位置）。 |
| 2 | 022 | `022-cifar10-distillation` | 1 个 / 355 行 | CPU* | 需 GPU | 自检已实测；完整对照未实测 | distillation | 020、021（先看量化，再看蒸馏；两者都在压缩模型）。 |
| 2 | 023 | `023-gap-vs-fc-distillation` | 1 个 / 92 行 | CPU | CPU 可跑 | 已实测通过 | params-vs-flops | 无。结构对照，不需要 GPU。 |
| 2 | 026 | `026-csr-spmv` | 1 个 / 186 行 | 特殊 | 需 GPU | 未实测 | csr-sparse | 025（CSR 的由来；025 只有片段，读概念即可）。 |
| 3 | 028 | `028-trt-benchmark` | 1 个 / 246 行 | 特殊 | 需 GPU | 未实测（需 TensorRT 10.x + NVIDIA GPU） | tensorrt-build、latency-breakdown | 027（推理链路的概念：为什么要量化/编译）。 |
| 3 | 029 | `029-trt-pipeline` | 1 个 / 267 行 | 特殊 | 需 GPU | 未实测（需 TensorRT 8.x + NVIDIA GPU） | tensorrt-build、latency-breakdown | 028（同一类实验，但 API 版本不同）。 |
| 4 | 039 | `039-on-device-npu` | 无代码 | GPU | 不适用 | 无 expected_output | — | 038（INT8 量化的概念）。 |
| 5 | 042 | `042-cuda-streams-concurrency` | 1 个 / 100 行 | CPU* | 需 GPU | 未实测 | async-copy | 003（CPU 与 GPU 的并发编排）。 |
| 5 | 043 | `043-vram-contention` | 1 个 / 94 行 | CPU* | 需 GPU | 未实测 | memory-accounting、occupancy | 042（Stream 与并发）。 |
| 5 | 044 | `044-gpu-task-dispatch` | 1 个 / 65 行 | 无 | CPU 可跑 | 已实测通过 | scheduling | 043（多模型同机的争用）。 |
| 5 | 047 | `047-nccl-allreduce` | 1 个 / 66 行 | CPU* | 需 GPU | 未实测（需 2 张 NVIDIA GPU + torchrun） | nccl | 046（为什么要对答案；046 只有片段）。 |
| 5 | 048 | `048-tensor-parallel-linear` | 2 个 / 123 行 | CPU* | 需 GPU | tp_split_check.py 已实测；tp_linear_d… | tp-parallel、nccl | 047（集合通信原语）。 |
| 5 | 051 | `051-parallel-group-check` | 1 个 / 92 行 | 无 | CPU 可跑 | 已实测通过 | distributed-3d | 048（一层 TP 的切法）、049/050（PP 与显存账本，读概念）。 |
| 6 | 056 | `056-custom-op-vs-eager` | 4 个 / 186 行 | CPU* | 需 GPU | 未实测（需 NVIDIA GPU + nvcc + torch） | fusion、dsl-torch、launch-overhead | 053–055（自定义算子的最小流程、接入 PyTorch、梯度）。 |
| 7 | 060 | `060-tensor-memory-ledger` | 1 个 / 88 行 | CPU* | 需 GPU | 未实测 | memory-accounting | 043（显存争用的现象）。 |
| 7 | 061 | `061-coalescing-lab` | 1 个 / 114 行 | 特殊 | 需 GPU | 未实测 | coalescing | 060（先知道显存怎么记账，再看访问方式）。 |
| 7 | 062 | `062-shared-memory-limit` | 1 个 / 115 行 | 特殊 | 需 GPU | 未实测（需 numba-cuda） | shared-memory、occupancy | 011（分块与同步）、061（合并访存）。 |
| 7 | 063 | `063-warp-broadcast` | 1 个 / 139 行 | 特殊 | 需 GPU | 未实测（需 numba-cuda） | constant-memory、coalescing | 061（合并访存）、062（共享内存的边界）。 |
| 7 | 064 | `064-host-to-device-wait` | 1 个 / 92 行 | CPU* | 需 GPU | 未实测 | pinned-memory、transfer | 061（访存是瓶颈的前提）。 |
| 7 | 065 | `065-async-transfer-overlap` | 1 个 / 127 行 | CPU* | 需 GPU | 未实测 | async-copy | 064（传输方式）、042（并发）。 |
| 7 | 066 | `066-nvlink-copy` | 1 个 / 101 行 | CPU* | 需 GPU | 未实测（需 2 张 NVIDIA GPU） | p2p-copy、transfer | 064、065（传输与重叠）。 |
| 8 | 067 | `067-onnx-export-alignment` | 1 个 / 126 行 | CPU | CPU 可跑 | 已实测通过 | onnx-export | 027（推理链路的概念）。 |
| 8 | 068 | `068-engine-build` | 1 个 / 69 行 | 特殊 | 需 GPU | 未实测（需 TensorRT 11.x + NVIDIA GPU） | tensorrt-build、fusion | 067（先有 ONNX，才能建引擎）。 |
| 8 | 069 | `069-engine-runtime` | 1 个 / 129 行 | 特殊 | 需 GPU | 未实测（需 TensorRT 11.x + 已构建的 engine） | tensorrt-runtime | 068（必须先有 engine）。 |
| 8 | 070 | `070-app-latency` | 1 个 / 123 行 | 特殊 | 需 GPU | 未实测（需 TensorRT 11.x + 已构建的 engine） | latency-breakdown、tensorrt-runtime | 069（运行时封装 run_engine 由本篇复用）。 |
| 8 | 071 | `071-int8-quant` | 1 个 / 64 行 | 特殊 | CPU 可跑 | 未实测（需 nvidia-modelopt） | int8-quant | 067（需要导出的 ONNX）。 |
| 8 | 072 | `072-pruning-kernel` | 2 个 / 203 行 | GPU | 需 GPU | 未实测 | pruning、transfer | 071、060（量化与显存记账）。 |
| 8 | 073 | `073-pruning-no-speedup` | 1 个 / 93 行 | CPU | CPU 可跑 | 已实测通过 | pruning、params-vs-flops | 072（先看清剪枝在算什么）。 |
| 9 | 074 | `074-ct-volume-basics` | 1 个 / 104 行 | CPU | CPU 可跑 | 已实测通过 | nifti-geometry | 无（医疗系列入口，只要会看 NIfTI 的 shape/affine）。 |
| 9 | 075 | `075-ct-seg-inference` | 2 个 / 114 行 | 特殊 | 需 GPU | 未实测（需数据 + MONAI 权重） | inference-pipeline、dice-metric | 074（体积与网格）。 |
| 9 | 076 | `076-ct-intensity-map` | 1 个 / 163 行 | GPU | 需 GPU | CPU 检查已实测通过；CuPy / CUDA 部分未实测 | fusion、transfer、dsl-cupy | 074（CT 强度与显存布局）。 |
| 9 | 077 | `077-sliding-window-toy` | 2 个 / 171 行 | 特殊 | 需 GPU | 部分实测（NumPy 教学版已实测，MONAI 连接检查未实测，逐… | sliding-window | 075（单窗口推理已跑通）。 |
| 9 | 078 | `078-ct-one-step-train` | 1 个 / 126 行 | 特殊 | 需 GPU | 已实测通过 | training-step、dice-metric | 075、077（会推理，也要会拼窗口）。 |
| 9 | 079 | `079-ct-amp-check` | 2 个 / 199 行 | 特殊 | 需 GPU | 格式检查已实测通过；AMP 部分未实测 | amp | 078（一次训练更新的体检）。 |
| 9 | 080 | `080-ct-double-buffer` | 1 个 / 141 行 | CPU* | 需 GPU | 未实测 | async-copy、pinned-memory | 078、077（推理与窗口）。 |
| 9 | 081 | `081-ct-seg-eval` | 1 个 / 151 行 | CPU | CPU 可跑 | 已实测通过 | dice-metric | 075（模型已经产出预测）。 |

## 各章速览

- **第 1 章 GPU 编程基础**（`01-gpu-programming/`，001–019，收录 17 篇，已实测 2 篇）
- **第 2 章 模型压缩**（`02-model-compression/`，020–026，收录 4 篇，已实测 2 篇）
- **第 3 章 推理服务与部署**（`03-inference-serving/`，027–035，收录 2 篇，已实测 0 篇）
- **第 4 章 端侧推理**（`04-edge-inference/`，036–041，收录 1 篇，已实测 0 篇）
- **第 5 章 并行与分布式**（`05-parallel-distributed/`，042–051，收录 6 篇，已实测 3 篇）
- **第 6 章 自定义算子**（`06-custom-operators/`，052–059，收录 1 篇，已实测 0 篇）
- **第 7 章 访存与传输**（`07-memory-and-transfer/`，060–066，收录 7 篇，已实测 0 篇）
- **第 8 章 TensorRT 部署**（`08-tensorrt-deploy/`，067–073，收录 7 篇，已实测 2 篇）
- **第 9 章 医疗影像实战**（`09-medical-imaging/`，074–081，收录 8 篇，已实测 6 篇）

## 按机制反查

标签含义见下表；同一篇可带 2–3 个标签，所以各项之和大于篇数。

- **`thread-mapping`**（线程与数据的映射，7 篇）：001、002、005、006、007、008、017
- **`latency-breakdown`**（时延拆分，6 篇）：003、004、013、028、029、070
- **`transfer`**（主机↔设备传输，6 篇）：004、013、064、066、072、076
- **`sync`**（同步与双缓冲，5 篇）：010、011、014、018、019
- **`async-copy`**（异步与重叠，4 篇）：003、042、065、080
- **`fusion`**（算子融合，4 篇）：009、056、068、076
- **`coalescing`**（合并访存，3 篇）：002、061、063
- **`dice-metric`**（分割评估，3 篇）：075、078、081
- **`shared-memory`**（共享内存分块，3 篇）：011、015、062
- **`tensorrt-build`**（引擎构建，3 篇）：028、029、068
- **`launch-overhead`**（启动开销，2 篇）：014、056
- **`memory-accounting`**（显存记账，2 篇）：043、060
- **`memory-reuse`**（中间结果复用，2 篇）：008、009
- **`nccl`**（集合通信，2 篇）：047、048
- **`occupancy`**（占用率，2 篇）：043、062
- **`params-vs-flops`**（参数量 vs 计算量，2 篇）：023、073
- **`pinned-memory`**（页锁定内存，2 篇）：064、080
- **`pruning`**（剪枝与稀疏，2 篇）：072、073
- **`tensorrt-runtime`**（运行时与形状，2 篇）：069、070
- **`amp`**（混合精度（AMP），1 篇）：079
- **`boundary`**（边界条件，1 篇）：019
- **`constant-memory`**（常量内存广播，1 篇）：063
- **`csr-sparse`**（CSR 稀疏格式，1 篇）：026
- **`data-race`**（数据竞争，1 篇）：017
- **`distillation`**（知识蒸馏，1 篇）：022
- **`distributed-3d`**（3D 并行，1 篇）：051
- **`dsl-cupy`**（CuPy 抽象，1 篇）：076
- **`dsl-numba`**（Numba 抽象，1 篇）：021
- **`dsl-torch`**（PyTorch 抽象，1 篇）：056
- **`inference-pipeline`**（推理链路，1 篇）：075
- **`int8-quant`**（INT8 量化，1 篇）：071
- **`nifti-geometry`**（医学影像几何，1 篇）：074
- **`onnx-export`**（导出一致性，1 篇）：067
- **`p2p-copy`**（点对点拷贝，1 篇）：066
- **`reduction`**（归约/前缀和，1 篇）：010
- **`scheduling`**（请求调度，1 篇）：044
- **`sliding-window`**（滑窗推理，1 篇）：077
- **`stability`**（数值稳定性，1 篇）：018
- **`tp-parallel`**（张量并行，1 篇）：048
- **`training-step`**（训练一步，1 篇）：078
- **`weight-quant`**（权重存储量化，1 篇）：021
