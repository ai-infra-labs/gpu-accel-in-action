# 09-medical-imaging 实测输出（N 卡 runbook）

**状态：未实测。** 本文件是「待填 runbook」，不是空白模板：每个字段都来自代码里
真实存在的 `printf`，行数、列序、单位都对得上。**一个没跑过的数字都不许填**——
仓库硬规矩第一条就是这么写的。

## 环境（实测时逐项填）

| 项 | 值 |
|---|---|
| OS | 待实测 |
| GPU | 待实测（型号 + 显存） |
| Driver | 待实测（`nvidia-smi` 首行） |
| CUDA Toolkit | 待实测（`nvcc --version` 末行） |
| 计算能力 | 待实测（`sm_XX`，与 `-arch` 一致） |

## 一、Track A 主线：backprojection

编译（**以下命令都在 `projects/ct-reconstruction/` 目录下执行**）：

```bash
cd ops/backprojection
for v in v0_naive v1_optimized v2_industrial; do
  nvcc -O3 -arch=native $v.cu -o $v && ./$v
done
nvcc -O3 -arch=native test.cu -o test && ./test
```

固定规模 `nx=256 ny=256 nDet=384 nAngles=180`，50 次平均、5 次预热。

### v0_naive

```text
[MI] 规模 nx=256 ny=256 nDet=384 nAngles=180
[MI] v0 时间 待实测 ms（50 次平均，5 次预热）
[MI] v0 有效带宽 待实测 GB/s（按 nx*ny*nAngles*8 字节计）
[MI] v0 估算算力 待实测 GFLOP/s（按每样本 14 FLOP 计）
[MI] v0 中心体素 待实测，全图均值 待实测
```

### v1_optimized

```text
[MI] 规模 nx=256 ny=256 nDet=384 nAngles=180
[MI] v1 时间 待实测 ms（50 次平均，5 次预热）
[MI] v1 有效带宽 待实测 GB/s（按 nx*ny*nAngles*8 字节计）
[MI] v1 估算算力 待实测 GFLOP/s（按每样本 14 FLOP 计）
[MI] v1 中心体素 待实测，全图均值 待实测
```

### v2_industrial

```text
[MI] 规模 nx=256 ny=256 nDet=384 nAngles=180
[MI] v2 时间 待实测 ms（50 次平均，5 次预热）
[MI] v2 有效带宽 待实测 GB/s（按 nx*ny*nAngles*8 字节计）
[MI] v2 估算算力 待实测 GFLOP/s（按每样本 14 FLOP 计）
[MI] v2 中心体素 待实测，全图均值 待实测
```

### test（正确性）

```text
[MI] golden(v0) 均值 待实测，非有限值 0 个
[MI] v1 vs golden: 超差 0 / 65536，最大相对误差 待实测
[MI] v2 vs golden: 超差 0 / 65536，最大相对误差 待实测
[MI] 判定：PASS（容差 1e-5，golden = v0）
```

`超差 0` 与 `判定 PASS` 是硬要求；`最大相对误差` 应远小于 `1e-5`（量级 `1e-7`）。
如果 v2 的最大相对误差逼近 `1e-5`，说明有人改了算法而不只是改了写法。

### 三项一致性检查

三版跑同一个规模，这两行必须几乎相同：

- `中心体素`：三版应在小数点后 4 位一致
- `全图均值`：三版应在小数点后 5 位一致

对不上就说明三版跑的不是同一个问题（常见原因：`-arch` 不同导致 `cosf` 精度不同，
或某个版本被改了采样边界）。

## 二、Track A 过渡算子

### ramp-filter

```bash
cd ops/ramp-filter && nvcc -O3 -arch=native v0_naive.cu -o v0_naive && ./v0_naive
```

```text
[MI] 规模 nAngles=180 nDet=384，block=128，shared=1544 字节
[MI] 与 CPU 参考最大绝对误差 待实测
[MI] 直流段滤波后最大幅值 待实测（应接近 0）
[MI] 判定：待实测（容差 1e-6）
```

`shared` 那一行是确定值：`(384+2)×4 B = 1544 B`。若打印出的不是 1544，
说明加载逻辑或规模被改了。

### bilinear-interp

```bash
cd ops/bilinear-interp && nvcc -O3 -arch=native v0_naive.cu -o v0_naive && ./v0_naive
```

```text
[MI] 规模 64x64，取样点 4096，block=128
[MI] GPU 与 CPU 最大绝对误差 待实测
[MI] 整数坐标处最大偏差 待实测（应接近 0）
[MI] 判定：待实测（容差 1e-6）
```

## 三、Track B 端到端（工程示例）

```bash
# 仍在 projects/ct-reconstruction/ 下
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./fdk_demo
```

预期输出格式（数值待实测）：

```text
[MI] fdk_demo (synth) nx=64 ny=64 nz=1 nDet=96 nAngles=64
[MI] FDK pipeline (synth) time: 待实测 ms
[MI] center row mean: 待实测 (count=64)
[MI] phantom center row mean: 待实测（参考值，非质量指标）
[MI] done
```

端到端耗时**不能**当作单算子优化收益，两者口径不同：

- Track B 计时包含 `cudaMemcpy` 上下行、滤波 kernel、反投影 kernel 与 `cudaDeviceSynchronize`；
- 端到端规模（64×64、96×64）远小于 Track A（256×256、384×180），
  连 kernel 启动开销占比都不同。

`center row mean` 是「重建确实跑出来了、不是全零」的最简证据；
它没有 golden 可比，**不能**用它证明重建正确。

## 四、NCU 采集

按 [`nsight.md`](nsight.md) 的命令采 `DRAM Throughput / Global Load Efficiency /
L2 Hit Rate / Achieved Occupancy / Warp Execution Efficiency` 五项，填回
[`benchmark.md`](benchmark.md) 的规模敏感性表与本文档下方。

| 指标 | v0 | v1 | v2 |
|---|---:|---:|---:|
| DRAM Throughput (%) | 待实测 | 待实测 | 待实测 |
| Global Load Efficiency (%) | 待实测 | 待实测 | 待实测 |
| L2 Hit Rate (%) | 待实测 | 待实测 | 待实测 |
| Achieved Occupancy (%) | 待实测 | 待实测 | 待实测 |
| Warp Execution Efficiency (%) | 待实测 | 待实测 | 待实测 |
| 寄存器/线程（`-Xptxas -v`） | 待实测 | 待实测 | 待实测 |

## 五、填完之后要做什么

1. 把数值同步回 [`benchmark.md`](benchmark.md) 的两张表；
2. 若某个预期趋势被推翻（例如 v2 反而更慢），**不要改结论迁就代码**，
   先按 `nsight.md` 的判断顺序找原因，把原因写进博文的 ⑥ 段；
3. 博文 ③④⑤⑥ 段的代码节选必须从这三份 `.cu` 里逐字复制，
   本目录已登记在 `docs/maintaining/provenance.json` 的 `standalone` 段，
   按自己的构建与运行结果验收，不要求对上某篇教程（见 `CONTRIBUTING.md` 的「完整工程」小节）。
