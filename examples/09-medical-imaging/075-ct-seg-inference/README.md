# 075 · AI 怎样在 CT 里勾出一个器官

分割模型不是"看一眼就画出来"，而是一条带空间约束的流水线：预处理（重采样、
窗宽窗位、方向对齐）→ 滑窗推理 → 逆变换回原网格 → 保存为与原 CT 同网格的掩膜。

本篇不训练，只把 MONAI 官方脾脏 bundle（0.6.0）跑一遍，然后**在原始网格上**
核对输出的空间信息，并算一个前景 Dice。

## 前置

074（体积与网格）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `run_spleen_bundle.py` | 装配 bundle 配置、单病例推理、写 `run_manifest.json` | 博文代码块 #4 |
| `eval_spleen_output.py` | 三份体积同网格校验 + 前景 Dice（双方皆空时返回 `null`） | 博文代码块 #7 |

## 运行

```bash
python -m pip install -r ../../requirements/advanced.txt

# 1) 推理：需要事先下载 spleen_ct_segmentation 0.6.0 bundle
python run_spleen_bundle.py /path/to/spleen_ct_segmentation case_image.nii.gz --out case_output
# CPU 也可以：--device cpu（慢，但流程一致）

# 2) 核对：影像、参考标签、预测必须同网格
python eval_spleen_output.py case_image.nii.gz case_label.nii.gz case_output/case_image_pred.nii.gz
```

`run_spleen_bundle.py` 会先校验 bundle 的后处理链顺序是否为
`Activationsd → Invertd → AsDiscreted → SaveImaged`——顺序不对就直接报错，
不会"跑出个结果来再说"。

## 环境要求

- `torch` + `monai` + `nibabel`（`requirements/advanced.txt`）
- 需要：MONAI bundle 权重（首次需下载）、一个真实 CT 病例、对应的参考标签（仅评估用）
- `--device cuda:0` 需 N 卡；`--device cpu` 可跑但很慢

## 观察点

- 单病例 Dice 只是这一个病例；same_grid 必须是 true，否则比较无效。
- 双方皆空时 Dice 为 null，不是 1.0——这一条决定了空掩膜病例怎么算均值。

## 验证状态

**未实测（需数据 + 权重）。** 本机未装 torch/monai，仅完成语法检查。

两个脚本分别需要：bundle 目录与真实病例；三份同网格体积。

## 出处

- 博文：[75-AI怎样在CT里勾出一个器官.md](../../../blog/075-AI怎样在CT里勾出一个器官.md)
- 代码块 #4，76 行，md5 `b303a3af191db3a79492e8482d0f2c7b` → `run_spleen_bundle.py`
- 代码块 #7，38 行，md5 `0098231d30f1c63797886e7ffafa5404` → `eval_spleen_output.py`

代码逐字一致。**博文未指定文件名**，两个文件名为仓库新增。
