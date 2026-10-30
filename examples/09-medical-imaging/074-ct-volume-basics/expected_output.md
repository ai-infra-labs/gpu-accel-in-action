# 074 · 实测输出

**状态：已实测通过**（CPU，2026-10-28，**合成数据**）。

## 1. 环境

```
平台    ：macOS 27.0.1 (arm64, Apple M4)
Python  ：3.13
numpy / nibabel / matplotlib：2.4.x / 5.4.2 / 3.x
数据    ：合成 NIfTI 一对，32×40×24，dtype int16（影像）/ uint8（标签）
spacing ：1.5 / 1.5 / 3.0 mm
```

合成数据只用于验证脚本的读取、几何门控与出图路径，不是真实病例。

## 2. 运行结果

命令：`python check_nifti_pair.py ct_image.nii.gz ct_label.nii.gz --out ct_three_planes.png`

退出码 `0`，原始输出：

```
Saved: /tmp/ctfix/ct_three_planes.png; metadata: /tmp/ctfix/ct_three_planes.json
```

`ct_three_planes.json` 的关键字段：

| 字段 | 实测值 | 说明 |
|---|---|---|
| `original.shape` | `[32, 40, 24]` | 与文件一致 |
| `axes` | `["R", "A", "S"]` | 对角 affine 的默认轴方向 |
| `spacing_mm` | `[1.5, 1.5, 3.0]` | 与写入的 affine 一致 |
| `dtype` | `float32` | **显示用**的 dtype，不是文件里的 int16 |
| `array_MiB` | `0.1171875` | `32×40×24×4 B / 2^20`，即 float32 的字节数 |
| `selected_ijk` | `[10, 12, 8]` | 三层各自前景面积最大的切片，落在范围内 |

PNG 约 77 KB，三列分别是 Axial / Coronal / Sagittal，图题写着
"Reference mask overlay | teaching view, not a prediction"。

## 3. 判读方式

- **`dtype` 与 `array_MiB` 都是显示用 float32 的**，不是原始文件的 int16。
  想核对原始体积要自己按文件头算，别拿这两个字段当"文件多大"的证据。
- `spacing_mm` 若为 `[1,1,1]`，说明文件没写空间信息；本脚本不做重采样，
  只会照原样读出。
- `selected_ijk` 选的是"前景面积最大"的切片，不是中心切片——小目标可能因此
  被选到，这正是查看标注的目的。

## 4. 这份数据不能说明什么

- **不是分割结果**：叠加的是参考标签，不是模型预测。
- **不涉及 GPU**：本篇不测显存与 Kernel。
- **真实数据的轴方向需要人眼复核**：合成数据是对角 affine，看不出 RAS/LPS
  搞反的典型错误。
