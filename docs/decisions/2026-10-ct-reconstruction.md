# 09-medical-imaging 落地实施方案（双轨：教学单元 + 工程示例）

> **路径提示**：本文写于 2026-10 的重构前后，文中出现的 `01-…09-*/`、`docs/ATLAS.md`、
> `docs/PROVENANCE.json` 等是当时的路径。现行对应关系：章节目录 → `examples/<章>/`，
> 能力矩阵 → `docs/reference/capability-matrix.md`，机器清单 → `docs/maintaining/provenance.json`。


> 状态：已落地（代码 + 文档；N 卡实测待办，见 §10.4）  
> 原则：**用「单文件」讲优化思路，用「极简项目」示范工程组织**  
> 范围：Phase 4 · AI Kernel Engineering（第 09 篇）

## 1. 背景与目标

本篇要覆盖「医学影像（CT 重建，FDK 思路）」中常用的 CUDA 算子，目标是既要让读者**学会怎么改、怎么快**，又要让读者**知道工业里怎么串**。

但真实 CT 重建是庞大的工程，本方案采取「**教学单元（Track A） + 工程示例（Track B）**」双轨并行：

- **Track A（教学优先）**：以单算子为主线，用 `v0_naive -> v1_optimized -> v2_industrial` 三层递进，**每个版本都能用 `nvcc` 单文件直接编译运行**。
- **Track B（工程示范）**：用极简 CMake 工程将少量算子串成 `滤波 -> 反投影` 的最短流水线，仅用于展示「模块划分」「数据流」「职责边界」这几个工程思路。

## 2. 主线选型

| 算子 | 教学价值 | 复杂度 | 推荐度 | 说明 |
|---|---|---|---|---|
| **反投影（Backprojection，FDK）** | ★★★★★ | 中高 | **主线（强烈推荐）** | 访存极度发散、线程映射不直觉，特别适合讲「全局内存合并」「访存顺序」「原子操作是否必要」「数据布局」。 |
| **斜坡滤波（Ramp Filter，1D）** | ★★★★☆ | 中 | **过渡用** | 适合讲 `__shared__`、`__syncthreads()`、一维卷积，但优化空间较局限，不适合走完整三层递进。 |
| **双线性插值（Bilinear Interpolation）** | ★★★★☆ | 较低 | **过渡用** | 常用于反投影采样，适合单独最小示例说明思路即可。 |

**锁定主线：`backprojection`**。博文 ③④⑤⑥ 全部围绕它展开，其余两个算子仅用于 ①② 和 ⑦ 的过渡说明。

## 3. 目录结构

### 3.1 Track A：教学版（`大模型训练与推理加速实战/cuda-kernels/09-medical-imaging/`）

```text
大模型训练与推理加速实战/cuda-kernels/09-medical-imaging/
├── ops/
│   ├── backprojection/      # 主线算子（必做）
│   │   ├── README.md
│   │   ├── v0_naive.cu
│   │   ├── v1_optimized.cu
│   │   ├── v2_industrial.cu
│   │   └── test.cu
│   ├── ramp-filter/         # 过渡算子（最小示例）
│   │   └── v0_naive.cu
│   └── bilinear-interp/     # 过渡算子（最小示例）
│       └── v0_naive.cu
├── benchmark.md            # 主线单算子 Benchmark
├── nsight.md               # 主线单算子 Nsight 分析要点
└── expected_output.md      # N 卡实测模板（待实测）
```

### 3.2 Track B：工程版（`cuda/projects/ct-reconstruction/`）

```text
cuda/projects/ct-reconstruction/
├── CMakeLists.txt
├── README.md
├── apps/
│   └── fdk_demo.cu         # 唯一入口
├── cmake/                  # （预留）
├── data/synthetic/         # 合成数据占位（不存真实 DICOM）
├── include/mi/
│   ├── common.cuh          # cudaCheck、CUDA Event 计时
│   ├── geometry.h          # 极简几何参数
│   ├── kernels.cuh         # 算子声明
│   └── utils.h             # 合成数据生成声明
└── src/
    ├── pipeline.cu         # 滤波 -> 反投影 流水线
    ├── utils.cu            # Phantom / Sinogram 生成（解析方式）
    └── kernels/
        ├── backproj.cu     # 体素驱动反投影（教学近似）
        ├── filter.cu       # 1D 简化 Ramp-like 滤波
        └── interp.cu       # 双线性插值（抽取自教学思路）
```

## 4. 各文件设计说明

### 4.1 Track A：`ops/backprojection/`

| 文件 | 设计思路 | 教学目的 |
|---|---|---|
| `v0_naive.cu` | **体素驱动（Voxel-driven）**：对每个体素 `(ix,iy)` 遍历所有投影角度 `a`，用 `cos/sin` + 简化几何映射到探测器坐标，再用双线性采样累加。 | 建立反投影概念，暴露访存极度发散（同一 warp 访问的 sinogram 数据很可能不连续）。 |
| `v1_optimized.cu` | **优化线程映射 + 访存顺序**：使用 `dim3 block(16,16)` 更贴合二维体数据，整理索引计算，减少冗余变量，向「更连续访问」靠拢。 | 展示「改线程映射/索引」往往比「改算法」先见效，强化合并访问意识。 |
| `v2_industrial.cu` | **预计算 + 权衡**：预计算 `cos(a)/sin(a)` 表（`angle_cos[.]/angle_sin[.]`），减少每次重复 `cosf/sinf`（视规模择用），精简计算、评估寄存器压力与占用率的权衡。 | 展示工业思路中的**权衡**（边际递减、计算 vs 访存、寄存器 vs Occupancy）。 |
| `test.cu` | 以 `v0_naive` 的结果作为 **golden**，对比 `v1`、`v2` 的 `volume`，采用 `|a-b|/(|b|+1e-12) < 1e-5` 判定通过。 | 保持本专栏「三层递进」一致的正确性约定。 |
| `README.md` | 问题定义、线程映射、编译运行、已知限制（几何为教学近似）、数据量建议。 | 清晰的教学定位，避免误以为是生产级 FDK。 |

### 4.2 Track B：工程版核心模块

| 文件/模块 | 设计思路 | 说明 |
|---|---|---|
| `include/mi/geometry.h` | 极简结构体 `MiGeometry` + `mi_default_geo2d()` | 集中管理几何参数，避免到处传标量。默认 `nx=ny=64, nz=1, nDet=96, nAngles=64`。 |
| `include/mi/common.cuh` | `MI_CUDA_CHECK`、`mi_time_ms` | 统一错误检查和 Event 计时，沿用专栏常用风格。 |
| `include/mi/kernels.cuh` | 算子声明 `mi_ramp_filter1d`、`mi_backproject_voxel` | 模块边界清晰。 |
| `src/utils.cu` | **解析生成**：`mi_gen_phantom_host`（简化 Shepp-Logan 思路，圆形/椭圆叠加）、`mi_gen_sinogram_host`（极简射线积分近似） | **不引入 DICOM**，全内存合成数据，跑通端到端即可。 |
| `src/kernels/filter.cu` | 极简 1D 滤波（中心差分近似 ramp-like） | 仅为串联流程服务，展示「逐行/逐投影」处理思路。 |
| `src/kernels/interp.cu` | 双线性插值 `bilinear_sample`（device inline） | 抽象出插值函数，便于后续替换。 |
| `src/kernels/backproj.cu` | 体素驱动反投影 kernel + `mi_backproject_voxel` 封装 | 从教学思路抽取**核心逻辑**，去掉大段教学注释，保留索引计算主干。 |
| `src/pipeline.cu` | `mi_run_fdk_demo`：生成数据 -> 滤波 -> 反投影，用 CUDA Event 计时，打印中心行均值做最简校验 | 管线化串联，展示数据流。 |
| `apps/fdk_demo.cu` | 唯一入口，打印参数并调用 pipeline，输出耗时与校验信息 | 极简 main()。 |
| `CMakeLists.txt` | `cmake_minimum_required(3.18)` + `project(... LANGUAGES CXX CUDA)`，`add_executable(fdk_demo ...)`，`CUDA_SEPARABLE_COMPILATION ON`，`CMAKE_CUDA_ARCHITECTURES native` | 极简且可观摩，避免复杂宏。 |

### 4.3 配套文档

| 文件 | 说明 |
|---|---|
| `benchmark.md` | 强制使用**主线单算子** `backprojection` 数据：`Time(ms) / Bandwidth(GB/s) / TFLOPS / Speedup`。表格对比 `v0/v1/v2`，并简述优化点与预期。 |
| `nsight.md` | 分析要点：`DRAM Throughput`、`Global Load Efficiency`、`L2 Hit Rate`、`Achieved Occupancy`、`Warp Execution Efficiency`。重点说明「访存发散如何体现在指标上」以及 `v0->v1->v2` 预期变化。 |
| `expected_output.md` | 模板：环境（OS/GPU/Driver/CUDA）、编译命令、问题规模、三版耗时与加速比、NCU 关键指标、端到端（工程版）耗时、校验信息。**标注「待 N 卡实测」**，不虚构精确数字。 |

## 5. 编译与验证

### 5.1 Track A（单文件，N 卡环境）

```bash
cd 大模型训练与推理加速实战/cuda-kernels/ct-reconstruction/ops/backprojection

# 编译运行
nvcc -O3 -arch=native v0_naive.cu -o v0_naive && ./v0_naive
nvcc -O3 -arch=native v1_optimized.cu -o v1_optimized && ./v1_optimized
nvcc -O3 -arch=native v2_industrial.cu -o v2_industrial && ./v2_industrial

# 正确性校验（golden = v0）
nvcc -O3 -arch=native test.cu -o test && ./test
```

过渡算子示例：
```bash
cd ../ramp-filter && nvcc -O3 -arch=native v0_naive.cu -o v0_naive && ./v0_naive
cd ../bilinear-interp && nvcc -O3 -arch=native v0_naive.cu -o v0_naive && ./v0_naive
```

### 5.2 Track B（工程版，N 卡环境）

```bash
cd cuda/projects/ct-reconstruction
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./apps/fdk_demo
```

预期输出示例（教学用）：
```text
[MI] fdk_demo (synth) nx=64 ny=64 nz=1 nDet=96 nAngles=64
[MI] FDK pipeline (synth) time: xxx ms
[MI] center row mean: xxx.xxxxxx (count=64)
[MI] done
```

> **本机（Apple Silicon 无 nvcc）只能做静态审查、结构整理和文档维护。所有实测数据必须在 N 卡环境完成后填入 `expected_output.md`。**

## 6. 博文 9 段用法（代码节选策略）

| 段落 | 用哪条轨道 | 建议做法 | 说明 |
|---|---|---|---|
| **① 为什么重要** | 原理 | 流程图：投影 -> 滤波 -> 反投影 | 用数据流图讲清应用场景，不贴代码。 |
| **② CPU vs GPU 思维** | 原理 | 对比串行 vs 并行思路 | 强调「体素/像素海量独立」但「访存极度发散」的矛盾。 |
| **③ 最小实现** | **Track A：v0_naive.cu** | 用 `__global__` + 极简 `main()` 核心段 | **必须用单文件**。讲透线程映射、几何映射、累加即可，省略冗长初始化。 |
| **④ 性能优化版本** | **Track A：v1/v2 节选** | **并列对比「改动前 vs 改动后」**，只截取改动那一小段（索引/循环/预计算） | 每版只突出 **1~2 个改动点**，其余用文字概括。 |
| **⑤ Benchmark** | **Track A：单算子实测** | 用 `Time(ms)/Bandwidth(GB/s)/TFLOPS/Speedup` 对比 v0/v1/v2 | **严禁用整条 pipeline 端到端替代单算子收益**。若边际递减要明确说明。 |
| **⑥ Nsight 分析** | **Track A：NCU** | 聚焦 `DRAM Throughput`、`Global Load Efficiency`、`L2 Hit Rate`、`Achieved Occupancy` | 用指标变化解释「为什么快/为什么没那么快」。 |
| **⑦ 工业应用** | **Track B：极简节选** | 仅节选 `include/mi/kernels.cuh`、`src/pipeline.cu`、`apps/fdk_demo.cu` 共 **10~20 行核心** | 讲「模块边界」「数据流」「职责划分」即可。**完全不用**贴 `CMakeLists.txt` 全文。 |
| **⑧ 面试题** | **Track A 思路** | 聚焦「体素驱动 vs 投影驱动」「访存发散」「原子是否必要」「索引计算」 | 用单算子逻辑作答，贴最短片段。 |
| **⑨ 延伸阅读** | **Track B 引导** | 指向 `cuda/projects/ct-reconstruction/README.md`，并**刻意引导回 `ops/backprojection/` 单文件** | 明确「工程示例」定位，避免读者从 CMake 起步。 |

## 7. 实施清单（落地顺序）

| 优先级 | 任务 | 状态 | 说明 |
|---|---|---|---|
| **P0** | 创建 Track A 目录骨架（`ops/backprojection/`、`ramp-filter/`、`bilinear-interp/`） | ✅ 已完成 | 已落地 |
| **P0** | 编写 Track A `backprojection` 三版 + `test.cu` + `README.md` | ✅ 已完成 | `v0/v1/v2/test/README` 全部就位 |
| **P0** | 编写 Track A 过渡算子 `ramp-filter/v0_naive.cu`、`bilinear-interp/v0_naive.cu` | ✅ 已完成 | 最小示例 |
| **P0** | 创建 Track B 工程骨架（`include/mi/`、`src/`、`apps/`、`CMakeLists.txt`） | ✅ 已完成 | 已落地 |
| **P0** | 实现 Track B 核心模块（`geometry/common/kernels/utils`、`filter/interp/backproj`、`pipeline`、`fdk_demo`） | ✅ 已完成 | 极简串联已实现 |
| **P0** | 编写 Track B `README.md`（强调教学优先路径） | ✅ 已完成 | 已加引导 |
| **P0** | 编写配套文档（`benchmark.md`、`nsight.md`、`expected_output.md`） | ✅ 已完成 | N 卡实测模板 |
| **P0** | 仓库契约适配：`pending` 登记 + `test_structure` 只认 `NNN-` 目录 + `make strict` 闸门 | ✅ 已完成 | 见 §10.2 |
| **P1** | 本机静态语法自检（clang CUDA 模式 + 最小假头，非实机编译） | ✅ 已完成 | 12 个 `.cu` 全部通过，`-Wall -Wextra` 无警告 |
| **P1** | 几何自检（NumPy 等价实现，对镜像方向） | ✅ 已完成 | 见 §10.3 |
| **P1** | `test.cu` 的 CPU 逐线程仿真执行（三版一致性断言） | ✅ 已完成 | PASS，零超差；见 §10.3 |
| **P1** | N 卡环境编译验证（Track A 单文件 + Track B 工程） | ⏳ 待实测 | 本机无 `nvcc` |
| **P1** | 填写 `expected_output.md`（耗时、带宽、加速比、NCU 指标） | ⏳ 待实测 | 实测后再填充，不虚构 |
| **P2** | 博文中按节选策略应用（③用v0、④用v1/v2节选、⑦用工程版节选） | ⏳ 待创作博文时应用 | 严格遵守节选策略 |

## 8. 设计约束（硬性准则）

- **单文件底线**：Track A 任意版本都必须能 `nvcc -O3 xxx.cu -o xxx && ./xxx` 独立跑通。破坏这点即视为失败。
- **主线收窄**：整篇优化叙事只围绕 `backprojection` 展开。其余算子绝不能抢占主线篇幅。
- **极简工程**：Track B **宁可显得简陋，也不要库化**。不要用抽象类、工厂、配置解析器。函数调用串联即可。
- **合成数据**：全程用解析方式生成 Phantom/Sinogram。**绝对不要**引入 DICOM 解析、ITK、DCMTK。
- **单算子计量**：Benchmark 必须用**单文件版**数据。**绝对不要**用整条 pipeline 的端到端耗时替代单算子优化收益。
- **教学优先引导**：Track B 的 `README.md` 必须显式指引回 Track A，防止读者折腾 CMake 而忘记学 Kernel。
- **不虚构实测**：`expected_output.md` 只写模板，实测前标注「待 N 卡实测」，实测后如实记录（允许环境差异导致数值不同）。
- **抽取而非复制**：Track B 从 Track A 抽取**核心 kernel**，可精简变量名和教学长注释，但必须保留关键索引计算逻辑，便于读者对照。

## 9. 快速自检

- [x] `projects/ct-reconstruction/` 结构完整且极简
- [x] `ct-reconstruction/ops/backprojection/` 有 `v0/v1/v2/test/README`
- [x] 过渡算子 `ramp-filter/`、`bilinear-interp/` 已就位
- [x] 配套 `benchmark.md`、`nsight.md`、`expected_output.md` 已创建
- [x] Track B `README.md` 含「教学优先路径」引导
- [x] 全程未引入医疗文件格式解析库
- [x] 全程采用合成数据
- [x] 主线锁定 `backprojection`
- [ ] N 卡实测（编译 + 运行 + NCU），结果回填 `expected_output.md`

## 10. 落地记录（与方案的偏差，逐条说明）

### 10.1 代码层面的三处调整

| 方案原写法 | 实际落地 | 理由 |
|---|---|---|
| v1 用 `block(16,16)` | `block(32,8)` | 16×16 会把一个 warp 劈成两行各 16 列，同一条投影行上只覆盖 16 个相邻探测器；32×8 让 `threadIdx.x` 与探测器轴对齐，warp 内 32 个取样点连续。这个改动本身可以在 `ncu` 里用 Global Load Efficiency 验证 |
| sinogram 用「极简射线积分近似」 | 椭圆 Radon 变换的**解析解** | 解析解没有离散化噪声，三版 kernel 的差异只可能来自 kernel 本身；离散近似会把噪声混进对比 |
| Track B 无 `pipeline.h` | 新增 `include/mi/pipeline.h` | `fdk_demo.cu` 与 `pipeline.cu` 分属两个编译单元，没有声明无法调用；9 行头文件，不违反「不库化」 |

### 10.2 仓库硬契约的适配（三处改动）

仓库原本的规矩是「仓库代码 = 博文代码块的逐字复刻」，而本方案是**代码先行**。
处理方式不是豁免测试，而是把「还没写博文」变成一等状态：

| 改动 | 文件 | 作用 |
|---|---|---|
| `pending` 登记段 | `docs/PROVENANCE.json` | 按目录前缀登记待博文代码，附 `planned_article` 与 `why` |
| 溯源脚本支持 pending | `scripts/build_from_blocks.py` | 扫描范围扩到全仓代码文件；未溯源且未登记即失败；`--strict` 把 pending 当失败 |
| 篇目结构测试只看 `NNN-` 目录 | `tests/test_structure.py` | `ops/` 不参与 81 篇的进度表校验，但仍受 pending 登记约束 |

另外新增 `tests/test_provenance.py::test_pending_entries_are_well_formed`
（登记必须写清篇目与理由、前缀必须以 `/` 结尾且其下真有文件），
`make strict` 作为发布闸门。**pending 不是免检，是一笔明账。**

### 10.3 本机做过的三项非实机验证

1. **静态语法自检**：用 clang 的 CUDA 模式（`--cuda-host-only -nocudainc`）配一份
   最小假 `cuda_runtime.h`，对 12 个 `.cu` 做 `-fsyntax-only -Wall -Wextra`。
   结果全部通过、无警告。它能抓语法与类型错误，**抓不到**链接、`<<<>>>` 运行语义、
   设备端行为——那些只有 N 卡能验。
2. **几何自检**：把同一套公式用 NumPy 重写，在 64×64 / 96×64 规模上比对重建图与
   phantom 的相关性（0.8508），并确认它高于左右镜像（0.8353）与上下镜像（0.8364），
   证明 `p = x·cosθ + y·sinθ` 的映射方向没搞反。加上三抽头高通后相关性从 0.5349
   升到 0.7494，说明滤波那一步确实在起作用。**这是推算，不是 GPU 实测。**
3. **`test.cu` 的 CPU 逐线程执行**：把 `<<<grid, block>>>` 机械改写成「对每个线程
   依次调用一次 kernel」（`threadIdx/blockIdx` 用全局变量注入），其余源码一行不改，
   用 `clang++ -x c++` 编译执行。三个版本都是线程独立的纯计算 kernel，没有
   `__shared__` 交互，所以这个仿真的**数值语义与 GPU 一致**（并行次序除外）。
   实跑结果（256×256 / 384×180，65536 体素）：

   | 项 | 结果 |
   |---|---|
   | golden(v0) 均值 | 2.746212（与 NumPy 自检的 2.7466 吻合，互为交叉验证） |
   | v1 vs v0 | 超差 **0 / 65536**，最大相对误差 `3.09e-07` |
   | v2 vs v0 | 超差 **0 / 65536**，最大相对误差 `4.54e-07` |
   | 判定 | **PASS**（容差 1e-5），退出码 0 |

   这说明三版**在数值上确实是同一个算法**，v2 的「除法换乘倒数」只带来 `1e-7` 量级
   的末位差异。但它**不能**替代实机：不验证编译、不验证 `<<<>>>` 语义、不验证设备端
   `cosf` 精度、更不产出任何耗时数据。`ramp-filter` 与 Track B 的 `filter.cu` 用了
   `__shared__` + `__syncthreads()`，这套逐线程仿真**不适用**（顺序执行时 barrier 不成立），
   所以没有对它们做仿真。

### 10.4 待办（交给有卡的机器）

- Track A 三版编译运行 + `test.cu` PASS；
- Track B `cmake` 构建 + `./fdk_demo` 输出；
- `ncu` 采五项指标；
- 全部填进 `expected_output.md` 与 `benchmark.md`，再把数值写进博文 ⑤⑥ 段。

## 附录：文件清单（实物落地）

### Track B（仓库根 `projects/ct-reconstruction/`）
- `CMakeLists.txt`
- `README.md`
- `apps/fdk_demo.cu`
- `include/mi/common.cuh`
- `include/mi/geometry.h`
- `include/mi/kernels.cuh`
- `include/mi/pipeline.h`
- `include/mi/utils.h`
- `src/pipeline.cu`
- `src/utils.cu`
- `src/kernels/backproj.cu`
- `src/kernels/filter.cu`
- `src/kernels/interp.cu`
- `data/synthetic/README.md`

### Track A（`09-medical-imaging/`）
- `ops/backprojection/README.md`
- `ops/backprojection/v0_naive.cu`
- `ops/backprojection/v1_optimized.cu`
- `ops/backprojection/v2_industrial.cu`
- `ops/backprojection/test.cu`
- `ops/ramp-filter/v0_naive.cu`
- `ops/bilinear-interp/v0_naive.cu`
- `benchmark.md`
- `nsight.md`
- `expected_output.md`

