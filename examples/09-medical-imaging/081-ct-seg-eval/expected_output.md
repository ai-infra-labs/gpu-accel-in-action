# 081 · 实测输出

**状态：已实测通过**（CPU，2026-10-28，**合成数据**）。

## 1. 环境

```
平台    ：macOS 27.0.1 (arm64, Apple M4)
Python  ：3.13
numpy / scipy / nibabel：2.4.x / 1.x / 5.4.2
数据    ：2 例合成体数据，32×40×24，spacing 1.5/1.5/3.0 mm
```

合成数据由脚本外的 NumPy 生成，只用于验证评估器本身；它不是任何真实病例。

## 2. 运行结果

命令：`python evaluate_ct_masks.py manifest.csv report.json`

退出码 `0`，标准输出（紧凑摘要）：

```json
{"n_requested": 2, "n_ok": 2, "n_error": 0, "n_patients": 1, "n_dice_defined": 2,
 "dice_mean": 0.9705882352941176, "n_hd95_defined": 2, "hd95_mean_mm": 1.5,
 "n_prediction_empty": 0, "n_reference_empty_fp": 0, "n_both_empty": 0}
```

逐例结果（来自 `report.json`）：

| case | 参考体素 | 预测体素 | Dice | HD95 (mm) | 状态 |
|---|---:|---:|---:|---:|---|
| c001（预测 = 参考） | 960 | 960 | 1.0 | 0.0 | `both_nonempty` |
| c002（预测整体平移 1 层） | 960 | 1080 | 0.9411764705882353 | 3.0 | `both_nonempty` |

## 3. 独立的数值复核

用另一段 NumPy 代码直接从 NIfTI 算一遍：

```
caseA L 960 P 960 I 960 dice 1.0
caseB L 960 P 1080 I 960 dice 0.9411764705882353
```

与脚本输出逐位一致（`2I/(L+P)`）。HD95 = 3.0 mm 也与"沿 z 平移一层、
z 方向 spacing 3.0 mm"相符。

## 4. 判读要点

- **`n_dice_defined` 与 `n_ok` 可能不等**：参考为空的病例 Dice 记为 `null`，
  不计入均值。这是刻意的——把它当 0 分会拉低均值，当 1 分会伪造完美。
- **`n_reference_empty_fp` 是最危险的计数**：参考为空但预测非空，说明模型在
  无病灶处产生了假阳性。这个数单独公布，不允许被 Dice 均值掩盖。
- **退出码只看文件级错误**：`n_error > 0` 返回非 0。质量好不好，退出码不知道。
- **几何门控先于评分**：`shape` 或 `affine` 不一致时直接记 `error`，
  不会退化成"插值到同一网格再算"——那会得到一个看起来正常但不可比的数。

## 5. 这份数据不能说明什么

- **不是模型精度**：两个病例都是合成的，预测由人工构造（相同 / 平移一层）。
- **不是性能**：纯 CPU 评估器，不测推理耗时。
- **HD95 的口径要一起读**：报告里写的是 `6-neighbour boundary centres;
  max directional linear P95; mm`。换一种边界定义，数字就会变，不能跨论文直接比。
