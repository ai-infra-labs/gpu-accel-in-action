# 仓库结构说明（自动生成，勿手改）

由 `python3 scripts/gen_structure.py` 扫描仓库生成：目录与计数来自磁盘，脚本说明来自各自 docstring 的首句，章节划分来自 `scripts/corpus.py`。
改完目录或新增脚本后 `make gen`，`make check` 会在过期时失败。

## 一、一层看全

```
gpu-accel-in-action/
├── blog/            81 篇博文 + assets/ 配图（代码块的 md5 是溯源基准）
├── examples/        九章配套代码：examples/01-gpu-programming/…/09-medical-imaging/
├── projects/        可独立构建的完整项目（CT 重建线），按自己的构建验收
├── cuda_kernels/    加载层：按名字查找 / 加载 / 运行某一篇
├── scripts/         工具脚本：抽取、校验、派生文档
├── tests/           结构 / 溯源 / 静态 / 冒烟 / 包契约测试
├── requirements/    三层依赖（CPU / GPU / 特殊环境）
├── docs/            入门 / 参考 / 维护 / 决策记录（见 docs/README.md）
├── .github/         CI（无 GPU，只跑 make check）
└── 根文件           README / Makefile / CMakeLists / pyproject / 贡献与变更
```

篇目：**81** 篇博文，**53** 个代码目录，**63** 个代码文件。

## 二、顶层目录逐个说

| 目录 | 作用 | 能否手改 | 性质 |
|---|---|---|---|
| `blog/` | 81 篇博文的唯一事实源，文件名 `NNN-短名.md`；配图在 `blog/assets/` | 可以，但改代码块必须同步 examples/ 下的代码并重跑 `make check` | 手写 |
| `examples/` | 九章配套代码：`<章>/<篇号-短名>/`，章内只放有代码目录的篇目 | 可以，新增篇目走 CONTRIBUTING 的流程 | 混合（章索引 README 自动生成） |
| `projects/` | 可独立构建的完整工程（CT 重建线），不以是否发表过文章作为合格标准 | 可以，新项目须在 provenance.json 的 standalone 段登记构建入口 | 手写 |
| `cuda_kernels/` | 加载层：按名字找到/加载/运行某一篇的脚本，不复制任何源码 | 可以（改完全仓测试会校验模块名唯一） | 手写 |
| `scripts/` | 工具脚本：抽取、校验、派生文档生成 | 可以，新增脚本要登记到 `static_check.LOCAL_MODULES`（若被 import） | 手写 |
| `tests/` | 结构 / 骨架 / 溯源 / 静态 / 冒烟 / 包契约，纯标准库，无卡可跑 | 可以，新增不变量优先加在这里 | 手写 |
| `requirements/` | 三层依赖：CPU 可跑 / 必须 GPU / 特殊环境（TensorRT、Triton、MPI） | 可以 | 手写 |
| `docs/` | 文档与派生看板：路线、能力矩阵、实测看板、构建、包说明、方案记录 | 部分：标「自动生成」的不要手改 | 混合 |
| `.github/` | CI：`make check`（无 GPU，只验证文档与代码对得上） | 可以 | 手写 |

## 三、三个代码区，规则不同

仓库里的代码分三类，**放在哪决定了它受什么约束**。混放是最常见的错误。

| 位置 | 是什么 | 溯源契约 | 构建方式 |
|---|---|---|---|
| `examples/<章>/<NNN-短名>/` | 某篇博文的逐字代码 | md5 必须命中 `blog/` 的代码块，或登记为拼接 | 单文件直接跑（多数是 PyCUDA 脚本） |
| `projects/ct-reconstruction/` | 可独立构建的完整工程 | 登记在 `docs/maintaining/provenance.json` 的 `standalone`；按自己的构建与测试验收 | `ops/` 单文件；`src|include|apps/` 走 CMake |
| `cuda_kernels/` | 加载层（工具，不是篇目代码） | 排除在溯源检查外，且不得复制篇目源码 | 不需要构建 |

判定口诀：**「这篇教程里有这段代码吗？」**有 → `examples/`；是一整个能独立构建的工程 → `projects/`；只是工具 → `scripts/` 或 `cuda_kernels/`。

三者**互不越界**：篇目代码不许脱离教程独自演化（md5 对账），完整工程不许拿「还没写文章」自我豁免（`built_by` 必须真实存在），工具不许复制任何篇目源码。

## 四、九章导航

| 章 | 目录 | 篇号范围 | 主题 | 这一章解决什么 | 篇目数 | 代码目录 | 代码文件 |
|---:|---|---|---|---|---:|---:|---:|
| 1 | `examples/01-gpu-programming/` | 001–019 | GPU 编程基础 | 线程→数据的映射、共享内存、同步 | 19 | 17 | 20 |
| 2 | `examples/02-model-compression/` | 020–026 | 模型压缩 | 参数量 / FLOPs / 显存三个量分开算 | 7 | 4 | 4 |
| 3 | `examples/03-inference-serving/` | 027–035 | 推理服务与部署 | 拆开一次推理请求的时间 | 9 | 2 | 2 |
| 4 | `examples/04-edge-inference/` | 036–041 | 端侧推理 | 端侧链路的真实约束 | 6 | 1 | 0 |
| 5 | `examples/05-parallel-distributed/` | 042–051 | 并行与分布式 | DP / PP / TP 怎么切、怎么验证 | 10 | 6 | 7 |
| 6 | `examples/06-custom-operators/` | 052–059 | 自定义算子 | 什么时候值得自己写算子 | 8 | 1 | 4 |
| 7 | `examples/07-memory-and-transfer/` | 060–066 | 访存与传输 | 访存与传输的六个机制 | 7 | 7 | 7 |
| 8 | `examples/08-tensorrt-deploy/` | 067–073 | TensorRT 部署 | ONNX → 引擎 → 运行时 | 7 | 7 | 8 |
| 9 | `examples/09-medical-imaging/` | 074–081 | 医疗影像实战 | 三维医疗影像的工程闭环 | 8 | 8 | 11 |

## 六、docs/ 按用途分区

文档按**读者要解决什么问题**分区，不按「是否自动生成」分区：

| 分区 | 放什么 | 谁看 |
|---|---|---|
| `docs/` 根 | 入口：`README.md` 导航 + 学习路线 + 快速开始 | 第一次来的人 |
| `docs/reference/` | 查阅型：文章索引、能力矩阵、验证状态、包说明 | 按需跳转的人 |
| `docs/maintaining/` | 维护者：仓库结构、示例约定、溯源契约与机器清单 | 要改仓库的人 |
| `docs/decisions/` | 决策记录：为什么这样组织、当时的取舍 | 半年后想改结构的人 |

标「自动生成」的不要手改，跑 `make gen` 重建：

| 文件 | 生成者 | 性质 |
|---|---|---|
| `docs/reference/capability-matrix.md` | `scripts/gen_atlas.py` | 自动生成 |
| `docs/reference/verification-status.md` | `scripts/gen_verify.py` | 自动生成 |
| `docs/reference/article-index.md` | `scripts/gen_index.py` | 自动生成 |
| `docs/_data/articles.json` | `scripts/gen_manifest.py` | 自动生成 |
| `docs/maintaining/repository-layout.md` | `scripts/gen_structure.py` | 自动生成 |
| `docs/LICENSE.md` | — | 手写 |
| `docs/README.md` | — | 手写 |
| `docs/decisions/2026-10-ct-reconstruction.md` | — | 手写 |
| `docs/decisions/2026-10-repository-restructure.md` | — | 手写 |
| `docs/getting-started.md` | — | 手写 |
| `docs/learning-path.md` | — | 手写 |
| `docs/maintaining/example-conventions.md` | — | 手写 |
| `docs/maintaining/provenance.json` | — | 手写 |
| `docs/maintaining/provenance.md` | — | 手写 |
| `docs/reference/package.md` | — | 手写 |

## 七、scripts/ 工具

| 脚本 | 一句话 |
|---|---|
| `build_from_blocks.py` | 校验（或重建）仓库代码与博文代码块的对应关系 |
| `corpus.py` | 仓库语料的单一数据源：扫描 blog/ 与九章目录，产出每篇的结构化字段 |
| `extract_block.py` | 从博文 Markdown 中抽取一个代码块，落成仓库里的源文件 |
| `gen_atlas.py` | 生成 docs/reference/capability-matrix.md：53 个篇目的能力矩阵 |
| `gen_chapters.py` | 生成九章目录的索引 README（`01-gpu-programming/README.md` … `09-medical-imaging/README.md`） |
| `gen_index.py` | 由 docs/_data/articles.json 生成 docs/reference/article-index.md |
| `gen_manifest.py` | 生成 docs/_data/articles.json：全仓库语料的机器可读清单 |
| `gen_structure.py` | 生成 docs/maintaining/repository-layout.md：仓库目录结构说明书 |
| `gen_verify.py` | 生成 docs/reference/verification-status.md：已落地篇目目录的「实测覆盖看板」 |
| `search.py` | 在仓库语料中检索篇目：按 id / 标题 / 标签(mechanisms) / 章节 / 状态 / 依赖 |
| `static_check.py` | 全仓静态检查：语法 + 依赖分层 + 硬件需求 |

## 八、tests/ 测试

全部只用标准库，无 GPU 可跑；缺第三方依赖的用例自动 skip。

| 模块 | 守什么 |
|---|---|
| `test_cpu_smoke.py` | 能在本机跑通的脚本做冒烟测试：退出码 0 + 关键输出行 |
| `test_manifest.py` | 清单与检索的校验 |
| `test_package.py` | cuda_kernels 包的契约测试：只用标准库，不需要 GPU |
| `test_provenance.py` | 代码与博文的溯源测试：逐字或拼接，两种来源都必须可验证 |
| `test_skeleton.py` | 骨架一致性：四个可机械判定的要素，每个文件都要有 |
| `test_static.py` | 静态检查：语法 + 依赖分层 + 硬件需求，全部在无 GPU 环境可跑 |
| `test_structure.py` | 仓库结构与进度表的一致性测试 |

## 九、根文件

| 文件 | 作用 |
|---|---|
| `README.md` | 定位、两个硬规矩、目录约定、验证链路、已收录清单 |
| `Makefile` | `check` / `gen` / `test` / `strict` / `build` / `search` 六个入口 |
| `CMakeLists.txt` | 单文件教学单元的构建（默认 `native` 架构，见 `docs/getting-started.md`） |
| `CODE_OF_CONDUCT.md` | 贡献者行为准则（改编自 Contributor Covenant 2.1） |
| `SECURITY.md` | 安全问题怎么报、哪些算安全问题、多久回应 |
| `pyproject.toml` | 打包配置：只打包 `cuda_kernels`，不把 81 篇脚本收进 wheel |
| `CONTRIBUTING.md` | 新增一篇 / 新增代码的流程与检查项 |
| `CHANGELOG.md` | 仓库级变更记录 |
| `LICENSE` | 代码 MIT |
| `docs/LICENSE.md` | 文档与博文 CC BY 4.0 |
| `.gitignore` | 忽略 `__pycache__`、打包产物与 `build/` |
| `.github/workflows/ci.yml` | push / PR 时跑 `make check` |

## 十、命名与放置约定（速查）

| 规则 | 正确 | 反例 |
|---|---|---|
| 篇目目录 = 三位篇号 + 短名 | `011-tiled-matmul-shared-memory/` | `tiled-matmul/` |
| 博文文件名 = 篇号 + 短名 | `blog/011-pycuda-tiled-matmul-….md` | `tiled-matmul.md` |
| 图 = `figure-<主题>-NN` | `blog/assets/figure-conv2d-01.jpg` | `blog/assets/orca-paste-1789634491.png` |
| 新增篇目代码放 examples/ | `examples/07-memory-and-transfer/061-…/` | 仓库根下 `061-…/` |
| 新代码必须可溯源 | 对上博文块 / 拼接登记 / pending | 直接丢进仓库任意位置 |
| 派生文档不手改 | 跑 `make gen` | 手改 `docs/reference/capability-matrix.md` |

## 附：完整目录清单

```
blog/  (直属 81 个文件；递归 0 个代码文件)
_meta/  (直属 3 个文件；递归 0 个代码文件)
assets/  (直属 59 个文件；递归 0 个代码文件)
cuda_kernels/  (直属 4 个文件；递归 5 个代码文件)
ops/  (直属 1 个文件；递归 1 个代码文件)
docs/  (直属 4 个文件；递归 0 个代码文件)
_data/  (直属 1 个文件；递归 0 个代码文件)
decisions/  (直属 2 个文件；递归 0 个代码文件)
maintaining/  (直属 4 个文件；递归 0 个代码文件)
reference/  (直属 4 个文件；递归 0 个代码文件)
examples/  (本层无文件；递归 63 个代码文件)
01-gpu-programming/  (直属 1 个文件；递归 20 个代码文件)
  001-fc-forward-backward/  (直属 3 个文件；递归 1 个代码文件)
  002-fc-backprop-sgd/  (直属 3 个文件；递归 1 个代码文件)
  003-cpu-gpu-cooperative/  (直属 3 个文件；递归 1 个代码文件)
  004-batch-input-pipeline/  (直属 3 个文件；递归 1 个代码文件)
  005-image-hflip/  (直属 3 个文件；递归 1 个代码文件)
  006-image-crop-roi/  (直属 3 个文件；递归 1 个代码文件)
  007-conv2d-correlation/  (直属 3 个文件；递归 1 个代码文件)
  008-conv3d-pycuda/  (直属 3 个文件；递归 1 个代码文件)
  009-fused-preprocess/  (直属 3 个文件；递归 1 个代码文件)
  010-reduction-prefix-sum/  (直属 4 个文件；递归 2 个代码文件)
  011-tiled-matmul-shared-memory/  (直属 4 个文件；递归 2 个代码文件)
  013-memory-transfer-matmul/  (直属 3 个文件；递归 1 个代码文件)
  014-bitonic-sort/  (直属 3 个文件；递归 1 个代码文件)
  015-naive-dft/  (直属 3 个文件；递归 1 个代码文件)
  017-nbody/  (直属 3 个文件；递归 1 个代码文件)
  018-heat-equation/  (直属 3 个文件；递归 1 个代码文件)
  019-fluid-grid/  (直属 4 个文件；递归 2 个代码文件)
02-model-compression/  (直属 1 个文件；递归 4 个代码文件)
  021-uint8-weight-quant/  (直属 3 个文件；递归 1 个代码文件)
  022-cifar10-distillation/  (直属 3 个文件；递归 1 个代码文件)
  023-gap-vs-fc-distillation/  (直属 3 个文件；递归 1 个代码文件)
  026-csr-spmv/  (直属 3 个文件；递归 1 个代码文件)
03-inference-serving/  (直属 1 个文件；递归 2 个代码文件)
  028-trt-benchmark/  (直属 3 个文件；递归 1 个代码文件)
  029-trt-pipeline/  (直属 3 个文件；递归 1 个代码文件)
04-edge-inference/  (直属 1 个文件；递归 0 个代码文件)
  039-on-device-npu/  (直属 1 个文件；递归 0 个代码文件)
05-parallel-distributed/  (直属 1 个文件；递归 7 个代码文件)
  042-cuda-streams-concurrency/  (直属 3 个文件；递归 1 个代码文件)
  043-vram-contention/  (直属 3 个文件；递归 1 个代码文件)
  044-gpu-task-dispatch/  (直属 3 个文件；递归 1 个代码文件)
  047-nccl-allreduce/  (直属 3 个文件；递归 1 个代码文件)
  048-tensor-parallel-linear/  (直属 4 个文件；递归 2 个代码文件)
  051-parallel-group-check/  (直属 3 个文件；递归 1 个代码文件)
06-custom-operators/  (直属 1 个文件；递归 4 个代码文件)
  056-custom-op-vs-eager/  (直属 6 个文件；递归 4 个代码文件)
07-memory-and-transfer/  (直属 1 个文件；递归 7 个代码文件)
  060-tensor-memory-ledger/  (直属 3 个文件；递归 1 个代码文件)
  061-coalescing-lab/  (直属 3 个文件；递归 1 个代码文件)
  062-shared-memory-limit/  (直属 3 个文件；递归 1 个代码文件)
  063-warp-broadcast/  (直属 3 个文件；递归 1 个代码文件)
  064-host-to-device-wait/  (直属 3 个文件；递归 1 个代码文件)
  065-async-transfer-overlap/  (直属 3 个文件；递归 1 个代码文件)
  066-nvlink-copy/  (直属 3 个文件；递归 1 个代码文件)
08-tensorrt-deploy/  (直属 1 个文件；递归 8 个代码文件)
  067-onnx-export-alignment/  (直属 3 个文件；递归 1 个代码文件)
  068-engine-build/  (直属 3 个文件；递归 1 个代码文件)
  069-engine-runtime/  (直属 3 个文件；递归 1 个代码文件)
  070-app-latency/  (直属 3 个文件；递归 1 个代码文件)
  071-int8-quant/  (直属 3 个文件；递归 1 个代码文件)
  072-pruning-kernel/  (直属 4 个文件；递归 2 个代码文件)
  073-pruning-no-speedup/  (直属 3 个文件；递归 1 个代码文件)
09-medical-imaging/  (直属 1 个文件；递归 11 个代码文件)
  074-ct-volume-basics/  (直属 3 个文件；递归 1 个代码文件)
  075-ct-seg-inference/  (直属 4 个文件；递归 2 个代码文件)
  076-ct-intensity-map/  (直属 3 个文件；递归 1 个代码文件)
  077-sliding-window-toy/  (直属 4 个文件；递归 2 个代码文件)
  078-ct-one-step-train/  (直属 3 个文件；递归 1 个代码文件)
  079-ct-amp-check/  (直属 4 个文件；递归 2 个代码文件)
  080-ct-double-buffer/  (直属 3 个文件；递归 1 个代码文件)
  081-ct-seg-eval/  (直属 3 个文件；递归 1 个代码文件)
projects/  (本层无文件；递归 14 个代码文件)
ct-reconstruction/  (直属 2 个文件；递归 14 个代码文件)
  apps/  (直属 1 个文件；递归 1 个代码文件)
  data/  (本层无文件；递归 0 个代码文件)
    synthetic/  (直属 1 个文件；递归 0 个代码文件)
  docs/  (直属 3 个文件；递归 0 个代码文件)
  include/  (本层无文件；递归 2 个代码文件)
    mi/  (直属 5 个文件；递归 2 个代码文件)
  ops/  (本层无文件；递归 6 个代码文件)
    backprojection/  (直属 5 个文件；递归 4 个代码文件)
    bilinear-interp/  (直属 1 个文件；递归 1 个代码文件)
    ramp-filter/  (直属 1 个文件；递归 1 个代码文件)
  src/  (直属 2 个文件；递归 5 个代码文件)
    kernels/  (直属 3 个文件；递归 3 个代码文件)
requirements/  (直属 3 个文件；递归 0 个代码文件)
scripts/  (直属 11 个文件；递归 11 个代码文件)
tests/  (直属 7 个文件；递归 7 个代码文件)
```
