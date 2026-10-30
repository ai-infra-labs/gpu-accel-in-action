# 060 · 实测输出

**状态：未实测。**

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
PyTorch / CUDA build：
分配器后端（get_allocator_backend）：
```

分配器后端要单独记：`native` 与 `cudaMallocAsync` 的缓存行为不同，
同一份代码在两种后端下的 reserved 曲线不一样。

## 2. 分阶段记账

| 阶段 | allocated (MiB) | reserved (MiB) | peak (MiB) |
|---|---|---|---|
| 参数加载后 | 待填 | 待填 | 待填 |
| 第 1 轮前向 | 待填 | 待填 | 待填 |
| `del` 中间结果后 | 待填 | 待填 | 待填 |
| `gc.collect()` 后 | 待填 | 待填 | 待填 |
| `empty_cache()` 后 | 待填 | 待填 | 待填 |

参数理论存储量：脚本会打印一行，用于核对 `allocated` 是否合理。

## 3. 判读

- `allocated` 降了、`reserved` 没降 → 显存回到缓存池，没还给驱动
- 两者都没降 → 还有引用存活（计算图、闭包、列表里的中间变量）
- `empty_cache()` 后 `reserved` 才降 → 属于正常，代价是下次分配要重新向驱动申请

## 4. 踩坑记录

待填。至少记录：

- `torch.cuda.memory_allocated` 与 `nvidia-smi` 的差值来源（CUDA context 固定开销）
- `synchronize` 缺失时数字为什么会出现偶发不一致
