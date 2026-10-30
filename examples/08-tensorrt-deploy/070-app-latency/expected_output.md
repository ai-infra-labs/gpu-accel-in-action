# 070 · 实测输出

**状态：未实测（需 TensorRT 11.x + 已构建的 engine）。**

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
TensorRT / torch / onnxruntime：
engine 路径 / batch：
```

## 2. 三种口径

脚本先打印两行字典（`measure_wall` 的统计），再打印一行事件法结果：

| 口径 | 含什么 | 待填值 |
|---|---|---|
| 每次创建张量（墙钟） | 分配 + H2D + 执行 + D2H + 同步 | 待填 |
| 复用缓冲区（墙钟） | H2D + 执行 + D2H + 同步 | 待填 |
| 事件法（10 组 × 200 次） | 只有引擎执行 | 待填（ms/次） |

判读方式：

- **三个数字的差值就是答案**：
  `每次创建 − 复用` ≈ 分配开销；`复用 − 事件法` ≈ 两次拷贝 + 同步开销。
  这正是"推理 1 ms、应用等 5 ms"的分解。
- 两个墙钟字典的键是 `samples / mean_ms / p50_ms / p95_ms /
  serial_samples_per_second`，逐项抄下来即可。**墙钟口径下 `p95` 才是应用关心的**，
  只看 `p50` 会低估抖动的体感；`serial_samples_per_second` 是按 batch 折算的串行完成率。
- **事件法测不到 H2D/D2H**：它记录的是流区间内的内核执行，所以数字最小。
  想测拷贝要另加事件，或者干脆用墙钟。

## 3. 这份数据不能说明什么

- **不是吞吐量**：每次调用都同步，串行测量，没有覆盖并发场景。
- **不含预处理/后处理**：真实应用的图像解码、resize、NMS 都不在内。
- **单模型固定形状**：batch 可调但形状固定，动态 shape 的重编译开销没算。

## 4. 踩坑记录

待填。至少记录：

- `PYTHONPATH` 没指到 `069-engine-runtime` 时，`from run_engine import ...` 直接
  `ModuleNotFoundError`——这不是引擎问题，别去查 TensorRT 日志
- `y_host.copy_(..., non_blocking=True)` 之后必须同步才能读，否则读到的是旧数据
- 复用缓冲区时 `x_host` 要在计时外准备好；每轮都重新生成输入会把 CPU 时间算进去
