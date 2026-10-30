# backprojection 单算子 Benchmark

**状态：未实测。** 本机（Apple Silicon）无 `nvcc`，下表只有指标口径与待填字段，
一个数字都没有填。填表方式见 [`expected_output.md`](expected_output.md)。

## 口径（先定口径，再看数字）

只测 **反投影 kernel 本身**，不含 sinogram 生成、不含主机↔设备拷贝、
不含 Track B 的端到端流水线。用 CUDA Event 包住 `kIters` 次连发，
`ms = 总时长 / kIters`，前面另有 5 次预热不计入。

| 指标 | 计算式 | 为什么这么算 |
|---|---|---|
| Time (ms) | `elapsed / kIters` | 冷启动成本被预热吃掉，剩下的才是稳态 |
| 有效带宽 (GB/s) | `nx·ny·nAngles·2·4 B / ms` | 每个采样点做双线性要摸 2 个相邻 `float`；这是**逻辑访存量**，不是 DRAM 流量，L2 命中会让 DRAM 侧小得多 |
| 估算算力 (GFLOP/s) | `nx·ny·nAngles·14 / ms` | 每样本按 14 FLOP 计（投影 3 + 索引换算 2 + 双线性 9），是**估算**，精确值看 `ncu` 的 FLOP 计数 |
| Speedup | `T(v0) / T(vN)` | 相对 v0 的加速比，不是相对某个外部基线 |

> 带宽与算力两项都叫「估算」，因为它们依赖上面这套记账口径。
> 写文章时不要把它们当成硬件峰值利用率——**利用率要看 `nsight.md` 里的 DRAM Throughput**。

## 待填表（N 卡实测后填写）

固定规模：`nx=ny=256, nDet=384, nAngles=180`，采样点 `1.18e7`。

| 版本 | Time (ms) | 有效带宽 (GB/s) | 估算算力 (GFLOP/s) | Speedup | 相对 v0 的改动 |
|---|---:|---:|---:|---:|---|
| v0_naive | 待实测 | 待实测 | 待实测 | 1.00× | 基准 |
| v1_optimized | 待实测 | 待实测 | 待实测 | 待实测 | 2D 线程映射 / `__ldg` / 指针递进 / 少一次乘 |
| v2_industrial | 待实测 | 待实测 | 待实测 | 待实测 | 常量内存角度表 / `__launch_bounds__` / 角度循环展开 |

### 规模敏感性（可选，第二张表）

同一份 v2，只改规模，用来回答「这个 kernel 是卡在访存还是卡在计算」：

| 配置 | nx=ny | nAngles | 采样点 | Time (ms) | 有效带宽 (GB/s) |
|---|---|---:|---:|---:|---:|
| 小 | 128 | 90 | 1.47e6 | 待实测 | 待实测 |
| 默认 | 256 | 180 | 1.18e7 | 待实测 | 待实测 |
| 压 L2 | 512 | 360 | 9.44e7 | 待实测 | 待实测 |

`nAngles` 从 180 翻到 360，采样点翻倍，`sinogram` 体量也翻倍（`360×384×4 B ≈ 553 KB`）。
`256` 那一档的 sinogram 只有 `276 KB`，很可能整块躺在 L2 里；`512` 那一档开始
真正打到 DRAM。**这两档测出来的「带宽」不可直接横比**，横比要配 `nsight.md` 的 L2 Hit Rate。

## 预期趋势（💡推断，实测后以数据为准）

| 现象 | 推断原因 | 若实测不符，优先查 |
|---|---|---|
| v1 比 v0 快 | v0 每体素每角度 2 次三角函数 + 1 次除法；warp 内 32 个取样点分散在 32 个不同行 | 线程映射是否真的让 `threadIdx.x` 对齐探测器轴 |
| v1 的提速幅度 **小于** 直觉 | 三角函数在 v0 里被 compiler 部分提升，且 L2 已吸收大部分重复读 | `ncu` 看 Instruction Mix：是 compute-bound 还是 memory-bound |
| v2 相对 v1 只快一点（边际递减） | 剩下的三角函数已是每体素 180 次，占比不高；瓶颈转向显存 | 是否 `nAngles` 太小导致常量内存收益被摊薄 |
| 三版带宽都远低于 HBM 峰值 | 双线性取样按 2 个 `float` 记账，但实际每个采样点只利用一次，且随机落在不同行 | 记账口径本身偏乐观，看 DRAM Throughput 更实在 |

**不要用 Track B 的端到端耗时替代这张表。** 端到端里混进了 sinogram 生成、
两次 `cudaMemcpy`、滤波 kernel，三个版本的差异会被稀释成噪声。

## 复现

```bash
cd ops/backprojection
for v in v0_naive v1_optimized v2_industrial; do
  nvcc -O3 -arch=native $v.cu -o $v && ./$v
done
```

每版都会打印「时间 / 有效带宽 / 估算算力 / 中心体素 / 全图均值」五个字段，
与上面两张表的列一一对应。中心体素与全图均值三版应几乎相同（差异只来自浮点结合律），
这也是一眼能看出的「跑的是同一个算法」的证据。
