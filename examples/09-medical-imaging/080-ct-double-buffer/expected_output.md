# 080 · 实测输出

**状态：未实测。** 本机无 CUDA，脚本在 `torch.cuda` 检查处即退出。

## 1. 环境

```
GPU 型号：
Driver / CUDA Toolkit：
torch / numpy 版本：
roi / batches / repeats：
```

## 2. 运行结果

脚本把结果写成 JSON 并原样打印，需要填的字段：

| 字段 | 待填值 |
|---|---|
| `serial_ms_median_p10_p90` | 待填 |
| `pipeline_ms_median_p10_p90` | 待填 |
| `synthetic_output_check` | 应为 `passed`（两种编排输出必须 `allclose`） |
| `gpu` / `cuda_runtime` / `torch` | 待填 |
| `warmup_rounds` | 应为 3 |

手工填一张对照表：

| 编排 | median (ms) | p10 | p90 |
|---|---|---|---|
| 串行（同流） | 待填 | 待填 | 待填 |
| 双缓冲 | 待填 | 待填 | 待填 |
| 加速比（串行/双缓冲） | 待填 | — | — |

判读方式：

- 加速比的理论上限由"拷贝时间 / 计算时间"决定：两者相等时最多 2×，
  计算远大于拷贝时趋近于 1×（重叠没意义）。
- 双缓冲没变快，先看三件事：主机内存是否 `pin_memory`（可分页内存无法异步拷贝）、
  两条流之间有没有多余的 `synchronize`、最后一个槽位有没有被下一批提前覆盖。
- `p10` 与 `p90` 的差距反映抖动。`repeats=5` 的样本量只够看趋势。

## 3. `--trace` 的产物

```
copy_compute_report_serial_trace.json
copy_compute_report_pipeline_trace.json
```

在 Chrome 的 `chrome://tracing` 或 Perfetto 里打开，看 CUDA 行上
`Memcpy HtoD` 与 `Kernel` 是否真的并排——这是"是否重叠"最直接的证据，
比总耗时更有说服力。

## 4. 这份数据不能说明什么

- **不是真实 CT 流水线**：合成窗口、2 层卷积的假模型，没有预处理、没有后处理。
- **不是端到端吞吐**：只测 H2D + 推理，D2H 与融合都不在内。
- **不涉及模型加载与权重传输**：那部分是一次性开销，不在本实验范围内。
