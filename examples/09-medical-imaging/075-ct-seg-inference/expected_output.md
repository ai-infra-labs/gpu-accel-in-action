# 075 · 实测输出

**状态：未实测（需数据 + MONAI 权重）。**
本机未装 torch / monai，只有一份真实 CT 与参考标签才能跑通。

## 1. 环境

```
平台 / Python：
torch / monai / nibabel 版本：
bundle 版本（应为 0.6.0）与权重 sha256：
设备（cuda:0 / cpu）：
病例来源：
```

## 2. 推理（`run_spleen_bundle.py`）

成功时只打印一行提示，真正的证据在 `run_manifest.json` 里：

| 字段 | 待填值 |
|---|---|
| `weights_sha256` | 待填 |
| `source_config_sha256` | 待填 |
| `torch` / `monai` / `cuda_runtime` | 待填 |
| `first_workflow_seconds` | 待填 |
| 输出 NIfTI 的形状与 affine | 待填 |

判读方式：

- `first_workflow_seconds` 包含权重加载与图构建，**不是稳态推理时间**。
  拿它跟论文里的单例耗时对比会得出错误结论。
- 若报 "后处理结构与本文核对的0.6.0配置不符"，说明 bundle 版本变了。
  这时不要去改代码里的 `expected` 列表硬凑——先确认新版本的后处理语义。
- 输出文件名带 `_pred` 后缀，且 `resample=False`：标签已经由 `Invertd` 回原网格，
  保存时不再二次插值。

## 3. 核对（`eval_spleen_output.py`）

输出是一段 JSON：

| 字段 | 待填值 | 说明 |
|---|---|---|
| `foreground_dice` | 待填 | 分母为 0（双方皆空）时为 `null`，不是 1.0 |
| `reference_voxels` | 待填 | 参考标签前景体素数 |
| `prediction_voxels` | 待填 | 预测前景体素数 |
| `same_grid` | 应为 `true` | 位置、形状、affine 三项都通过才为 `true` |

判读方式：脚本先做几何门控，再做 Dice。三份体积的 `shape` 与 `affine`
必须一致（`atol=1e-3`），否则直接报错——**不同网格上算 Dice 是无效的**，
哪怕数字看起来很正常。

## 4. 这份数据能说明什么

- 一条**可检查**的推理链路：权重、配置、设备、耗时都有摘要可追溯。
- 单病例的 Dice 只能说明这一个病例，不能外推成模型精度。

不能说明：分割质量是否达标（单病例）、GPU 利用率、吞吐量。
