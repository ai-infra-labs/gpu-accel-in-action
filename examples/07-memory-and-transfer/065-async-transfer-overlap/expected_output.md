# 065 · 实测输出

**状态：未实测。**

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
PyTorch / CUDA build：
batches / 输入形状 / dtype：
```

## 2. 三模式对比

| 模式 | 总耗时 (ms) | 相对 blocking |
|---|---|---|
| blocking | 待填 | 1.00× |
| same_stream | 待填 | 待填 |
| overlap | 待填 | 待填 |

判读方式：

- `same_stream` 与 `blocking` 接近 → 说明 `non_blocking` 没有创造重叠，
  只是把阻塞点挪到了后面
- `overlap` 明显低于 `blocking` → 拷贝与计算真的并行
- `overlap` 无改善 → 检查主机内存是否已 pin、Stream 间是否加了多余同步

## 3. 正确性

| 项 | 结果 |
|---|---|
| 各模式结果与 blocking 基线一致 | 待填 |

## 4. trace 时间线

脚本会写出 trace 文件路径。需要确认的事实（不能只看总耗时）：

- [ ] 拷贝条与计算条在时间轴上是否真的交叠
- [ ] 是否出现意料之外的同步点（把并行切断）
- [ ] 双缓冲的两个 slot 是否被正确交替使用

## 5. 踩坑记录

待填。至少记录：

- 首次迭代是否包含 CUDA context 初始化开销
- Event 记录顺序写错时，会出现"数字漂亮但结果错"还是"结果对但没重叠"
- trace 文件用什么工具打开（浏览器打开 `chrome://tracing` 或 Perfetto）
