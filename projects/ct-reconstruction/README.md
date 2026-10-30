# 09-medical-imaging · CT 重建（Track A 教学单元 + Track B 工程示例）

> **这是一个不属于 81 篇体系的独立内容线。** 本章的 074–081 是「CT 分割链路」八篇正篇，
> 这里讲的是另一件事：**CT 重建**（滤波 + 反投影）的算子优化与工程组织。
> 两者物理上分开存放：正篇在 `09-medical-imaging/`，本目录在仓库根 `projects/`。

```
projects/ct-reconstruction/
├── ops/          Track A：教学主线，单文件、nvcc 一把编、v0→v1→v2 三层递进
├── include/ src/ apps/   Track B：工程示例，CMake 工程，示范模块边界
└── docs/         benchmark / nsight / expected_output（单算子四指标 + NCU 要点）
```

> **想学 kernel 怎么写、怎么变快，先去 [`ops/backprojection/`](ops/backprojection/)。**
> 那边是单文件、三版递进的教学主线；本目录的 CMake 工程只回答另一个问题：
> **这些算子放进一个真项目里，边界怎么划。**

> 状态：**未实测**。本机无 `nvcc`，未在 NVIDIA GPU 上编译运行。

## 这个示例存在的理由

单文件教学程序有一个天然缺陷：所有东西都在一个文件里，读者学不到
「算子之间怎么拼」。而真实项目里，反投影不会自己跑起来——上游有滤波，
下游有落盘，中间有设备内存的分配与回收。

所以这里用**最小**的代价把这条链路串起来：

```
apps/fdk_demo.cu          唯一入口，只负责打印参数 + 调 pipeline
      ↓
src/pipeline.cu           生成数据 → 滤波 → 反投影 → 取一行做体检
      ↓                       ↑ 只通过设备指针交接，不知道 block 怎么划
src/kernels/filter.cu    src/kernels/backproj.cu
      ↘                       ↙
       src/kernels/interp.cu  （双线性取样，两个算子共用）
      ↓
src/utils.cu              合成 phantom / sinogram（不读任何文件）
```

## 目录

| 路径 | 内容 | 为什么单独一层 |
|---|---|---|
| `ops/backprojection/` | v0/v1/v2 + test + README（Track A 主线） | 单文件可独立编译，讲优化思路的地方 |
| `ops/ramp-filter/`、`ops/bilinear-interp/` | 两个过渡算子的最小示例 | 只讲一个点，不抢主线篇幅 |
| `docs/benchmark.md` | 单算子四指标（时间/带宽/算力/加速比） | 口径与填表规则 |
| `docs/nsight.md` | NCU 五项指标怎么读 | 「为什么快」的证据要求 |
| `docs/expected_output.md` | N 卡实测 runbook | 未实测时只给待填字段 |
| `include/mi/geometry.h` | `MiGeometry` + `mi_default_geo2d()` | 参数有单一来源，改规模只改一处 |
| `include/mi/common.cuh` | `MI_CUDA_CHECK` / `mi_time_ms` | 错误检查与计时全项目统一 |
| `include/mi/kernels.cuh` | 三个算子声明 | **模块边界**：pipeline 只认这份清单 |
| `include/mi/utils.h` | 合成数据生成声明 | 数据来源与算法实现分开 |
| `include/mi/pipeline.h` | `mi_run_fdk_demo` | 入口只需要一个函数声明 |
| `src/kernels/filter.cu` | 1D 斜坡滤波（简化） | 逐投影一行，shared memory 做邻居复用 |
| `src/kernels/interp.cu` | 双线性取样（设备端） | 抽出来，换取样策略只改一处 |
| `src/kernels/backproj.cu` | 体素驱动反投影 | 从 Track A 抽取主干 |
| `src/utils.cu` | phantom / sinogram 解析生成 | 没有 DICOM、没有 ITK |
| `src/pipeline.cu` | 串联 + 计时 + 体检 | 数据流的唯一描述处 |
| `apps/fdk_demo.cu` | `main()` | 极简，不含业务逻辑 |
| `data/synthetic/` | 合成数据说明（不存真实影像） | 全程内存生成，落到磁盘反而多一层依赖 |

## 构建与运行

```bash
cd projects/ct-reconstruction
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./fdk_demo
```

预期输出（数值待 N 卡实测）：

```text
[MI] fdk_demo (synth) nx=64 ny=64 nz=1 nDet=96 nAngles=64
[MI] FDK pipeline (synth) time: xxx ms
[MI] center row mean: xxx.xxxxxx (count=64)
[MI] phantom center row mean: xxx.xxxxxx（参考值，非质量指标）
[MI] done
```

`center row mean` 只能说明「重建确实算出来了、不是全零或 NaN」。
它**不是**精度指标——未滤波的反投影与 phantom 之间没有逐值对应关系。

## 三处值得对照的工程选择

**1. 算子之间只传设备指针。**
`mi_ramp_filter1d` 与 `mi_backproject_voxel` 的签名里没有 `MiGeometry` 之外的
耦合：pipeline 不需要知道 filter 用了多少 shared memory，也不需要知道
backproj 的 block 是 32×8 还是 16×16。换实现不动调用点。

**2. 取样函数被抽成独立编译单元，代价是内联丢失。**
`mi_bilinear_sample` 定义在 `src/kernels/interp.cu`，声明在头文件里，
依赖 `CUDA_SEPARABLE_COMPILATION`（可重定位设备代码）才能跨编译单元链接。
不加 `-dlto` 时它不会被内联——**这是模块边界的真实成本**。生产代码通常反过来做：
把这种小函数放进头文件 `__forceinline__`，牺牲边界换性能。示例里刻意保留了这个
「不划算但清晰」的版本，好在文中把权衡讲清楚。

**3. 计时口径写在代码里，不靠注释约定。**
`pipeline.cu` 的 Event 只包住两个 kernel，不含 H2D/D2H。这是刻意的：
端到端耗时里混进传输，会把算子优化效果稀释成噪声。要测端到端，另起一个窗口。

## 已知限制

1. **规模极小**（64×64、96×64）。为的是让端到端秒级跑完，不是为了性能。
   要看性能请去 Track A 的 256×256×180（`ops/backprojection/`）。
2. **滤波是简化版**。中心差分高通，只保留「去直流」的效果，不是频域 |ω| 滤波。
   它确实在起作用：NumPy 等价实现的几何自检（见 Track A 的 `ops/backprojection/README.md`）
   显示，加上这步后重建图与 phantom 的高频相关性从 0.53 升到 0.75，直流被压掉
   （均值 2.75 → 0.0006）。这些是推算值，不是 GPU 实测。
3. **只支持单层（nz=1）**。结构体里留了 `nz` 字段，但反投影只处理一层。
4. **合成数据**。物体是若干椭圆的叠加，sinogram 用解析线积分生成，
   没有真实 CT 的噪声、散射与射束硬化。
5. **没有落盘**。结果留在内存里，只打印中心行均值，不写 NIfTI。
   要可视化请用 `../../examples/09-medical-imaging/074-ct-volume-basics/` 的三平面视图脚本。

## 与 Track A 的对照

| Track A（教学） | Track B（工程） | 变化 |
|---|---|---|
| `v0_naive.cu` 一个线程一个体素 | `src/kernels/backproj.cu` | 同样的主干，去掉教学注释 |
| 单文件自带 `main()` 与计时 | `apps/` + `src/pipeline.cu` | 入口与流程分离 |
| 双线性写在 kernel 文件里 | `src/kernels/interp.cu` | 抽成独立算子 |
| 无滤波 | `src/kernels/filter.cu` | 补上前置步骤，形成链路 |
| 无合成数据生成 | `src/utils.cu` | 数据来源独立成层 |

**学顺序建议**：先 Track A 把 kernel 吃透，再看 Track B 学组织方式。
反过来先折腾 CMake，很容易在构建系统上消耗掉全部耐心，而 kernel 一行没学。
