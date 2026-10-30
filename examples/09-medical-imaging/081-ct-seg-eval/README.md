# 081 · CT 分割跑完了，怎样判断它做得好不好

Dice 谁都会算，坑在分母与前提：**不同网格上算 Dice 是无效的**；**双方都空时
Dice 没有定义**，不能悄悄记成 1.0；**参考为空但预测非空**是最危险的一种"高分"。

本篇是一个纯 CPU 的批量评估器：读清单、逐例做几何门控、再算 Dice 与 HD95，
并把失败病例留在总分母里。

## 前置

075（模型已经产出预测）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `evaluate_ct_masks.py` | CSV 清单 → 几何校验 → Dice/HD95 → JSON 报告 + 紧凑摘要 | 博文代码块 #2 |

## 运行

```bash
python -m pip install -r ../../requirements/python-cpu.txt
python evaluate_ct_masks.py manifest.csv report.json
```

清单格式（路径相对清单所在目录）：

```csv
case_id,patient_id,image,label,prediction
c001,p01,caseA_ct.nii.gz,caseA_seg.nii.gz,caseA_pred.nii.gz
```

- `case_id` 必须唯一，`patient_id` 可以重复（用于统计例数）
- 三份体积必须是 mm 单位的三维 NIfTI，且**同 shape、同 affine**（`atol=1e-3`）
- 退出码：`0` 表示全部病例评分完成；非 0 表示有文件级错误——**与分割质量无关**

## 环境要求

- `nibabel` + `scipy` + `numpy`（`requirements/python-cpu.txt`），CPU 即可
- 不执行推理，也不测 GPU 性能

## 观察点

- Dice 要与手算逐位一致（本仓用独立 NumPy 复核过 1.0 与 0.9412）。
- 参考为空时 Dice 记 null，不进均值；假阳性单独计数，不能被均值掩盖。

## 验证状态

**已实测通过**（CPU，2026-10-28，**合成数据**：2 例 32×40×24 的体数据，
spacing 1.5/1.5/3.0 mm）。Dice 另用独立 NumPy 手算复核，见 `expected_output.md`。

真实病例清单未验证。

## 出处

- 博文：[81-CT分割跑完了怎样判断它做得好不好.md](../../../blog/081-CT分割跑完了怎样判断它做得好不好.md)
- 来源代码块 #2，151 行，md5 `3cf7dc1f89fcb50069adca905edd183e`

代码逐字一致。博文原稿另附一份同名脚本（见 `blog/081-CT分割跑完了怎样判断它做得好不好.md` 文末），
与仓库这份 md5 相同（`3cf7dc1f89fcb50069adca905edd183e`），已核对。
