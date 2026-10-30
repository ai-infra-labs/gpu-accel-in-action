# backprojection 的 Nsight 分析要点

**状态：未实测。** 指标区只给「看哪几个、怎么读、期望什么方向」，
具体数值等 N 卡上采完再填。

## 采什么

```bash
# 单 kernel 全量指标（最慢，但信息最全）
ncu --set full -k regex:bp_v ./v1_optimized

# 只要关键几项，跑得快一些，适合三版横比
ncu --metrics \
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
  lts__t_sectors_op_read.sum,\
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  smsp__sass_thread_inst_executed_op_fp32_pred_on.sum \
  -k regex:bp_v ./v1_optimized
```

`--set full` 会重放 kernel 多次，180 个角度的循环会被反复跑，
采样时间比实际运行长一个量级，别在共享卡上无脑全量采。

## 五个必看指标

| 指标 | 衡量的东西 | v0 期望 | v1 期望 | v2 期望 | 怎么读 |
|---|---|---|---|---|---|
| **DRAM Throughput** | 显存带宽压到峰值的百分比 | 低 | 低-中 | 中 | 这是「访存墙」的唯一硬证据。若三版都 <10%，说明瓶颈不在这里，别在访存上继续雕花 |
| **Global Load Efficiency** | 一次请求里真正用上的字节占比 | **低**（warp 内 32 个取样点散在不同行） | 中 | 中 | 从 v0 到 v1 的跃升，就是「改线程映射先见效」的量化证据 |
| **L2 Hit Rate** | 二级缓存命中率 | 高 | 高 | 高 | 默认规模 sinogram 只有 276 KB，很可能整块在 L2 里。命中率高不等于没问题，它说明**这个规模测不出 DRAM 压力** |
| **Achieved Occupancy** | 实际驻留 warp 占理论上限的比例 | 待测 | 待测 | 待测 | v2 加了 `__launch_bounds__(256)` 与 `#pragma unroll 4`，寄存器用量上升可能压低占用率——若 v2 出现「占用率降了但更快」，说明延迟被指令级并行掩盖了 |
| **Warp Execution Efficiency** | 活跃线程占 warp 的均值比 | ≈100% | ≈100% | ≈100% | 这个 kernel 没有分支发散（只是边界判断），此值若明显低于 100%，先怀疑 `nx/ny` 不是 block 的整数倍导致的尾部空转 |

## 访存发散在指标上长什么样

反投影的取样地址 `u = x·cosθ + y·sinθ` 随线程变化，v0 里一个 warp 的 32 个线程
被一维摊平后落在不同行、不同列，探测器坐标互不相邻 → 一次 warp 的 32 次取样
可能拆成十几次内存事务，**Global Load Efficiency 会很低**。

v1 把 `threadIdx.x` 对齐体素列后，同一 warp 的 32 个线程在同一行、相邻 32 列，
`u` 也随之相邻，事务数接近理论下限。这条因果链是：

```
线程映射 → 取样地址是否连续 → 每 warp 的事务数 → Global Load Efficiency
```

写文章时按这条链讲，比只贴两张柱状图有用。

## 采完之后的判断顺序

1. 先看 **DRAM Throughput**：低于峰值 20% 就别再指望靠压缩访存提速，
   瓶颈在延迟或计算。
2. 再看 **Global Load Efficiency**：若 v0→v1 没明显改善，说明线程映射没生效
   （常见原因：`blockDim.x` 与 `nx` 的整除关系没对上，或索引算错再被夹取掩盖）。
3. 最后看 **Achieved Occupancy** 与寄存器数（`nvcc -Xptxas -v`）：
   v2 若因展开把寄存器顶到溢出（`spill`），性能会掉头向下——这正是
   `benchmark.md` 里「边际递减」要交代的那一层。

## 与 benchmark.md 的对应关系

| benchmark 里看到的 | nsight 里找答案 |
|---|---|
| v1 提速有限 | Global Load Efficiency 是否已到 90%+；DRAM Throughput 是否仍低 |
| v2 提速更小 | 寄存器/占用率变化；常量内存是否真有广播收益 |
| 带宽数字远低于峰值 | L2 Hit Rate —— 命中率高说明测的是缓存不是显存 |
