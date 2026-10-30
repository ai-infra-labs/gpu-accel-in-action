# 074 · CT 看着是一叠图，GPU 要处理的究竟是什么

一份胸部 CT 在屏幕上是"一叠切片"，在程序里却是一个三维数组加一套空间定义：
体素间距（spacing）、方向（affine）、单位（mm）。同一个数组配错 affine，
切片看起来没变，坐标却全错了。

本篇不训练、不推理，只做一件更基础的事：读入一对**同网格**的影像与参考标签，
核对几何一致性，再画三平面叠加图——图题明确写着"teaching view, not a prediction"。

## 前置

无（医疗系列入口，只要会看 NIfTI 的 shape/affine）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `check_nifti_pair.py` | 读 NIfTI、校验三维/单位/affine、选前景最大的切片画三平面图、导出元数据 JSON | 博文代码块 #4 |

## 运行

```bash
python -m pip install -r ../../requirements/python-cpu.txt
python check_nifti_pair.py case_image.nii.gz case_label.nii.gz --out ct_three_planes.png
```

三个参数：影像路径、同病例参考标签路径、输出图（默认 `ct_three_planes.png`）。
产出两张文件：PNG 叠加图与同名 `.json` 元数据（形状、affine、轴方向、spacing、
dtype、数组体积、所选切片号）。

## 环境要求

- `nibabel` + `matplotlib` + `numpy`（CPU 即可，脚本已设 `Agg` 后端）
- **需要真实的 NIfTI 数据**：脚本不接受合成输入，两个文件的网格必须一致

## 观察点

- axes 与 spacing 要和文件头对得上；反过来会让人以为「左肺跑到右边」。
- array_MiB 是显示用 float32 的字节数，不是原始 dtype 的体积。

## 验证状态

**已实测通过**（CPU，2026-10-28，**合成数据**：32×40×24、spacing 1.5/1.5/3.0 mm
的一对 NIfTI 影像与标签）。输出见 `expected_output.md`。

真实 CT 病例未验证——脚本对单位、三维形状、affine 的校验逻辑与数据来源无关，
但真实数据的轴方向与间距需要人眼复核。

## 出处

- 博文：[74-CT看着是一叠图-GPU要处理的究竟是什么.md](../../../blog/074-CT看着是一叠图：GPU要处理的究竟是什么.md)
- 来源代码块 #4，104 行，md5 `9fdbdf826d65b260ab4869bc2a36d843`

代码逐字一致。**博文未指定文件名**，`check_nifti_pair.py` 为仓库新增命名。
