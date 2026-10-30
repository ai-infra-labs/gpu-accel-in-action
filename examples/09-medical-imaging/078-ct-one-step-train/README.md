# 078 · 器官已经标出来了，AI 还要学什么

有了标签，下一步才是训练。但"能跑起来"和"训对了"是两件事：这一步只做**一次**
参数更新，然后检查四件事——损失是否有限、梯度是否非零且有限、参数是否变了、
变化是否有限。

本篇的模型是缩小的随机 UNet（通道 4/8/16），数据默认合成，故意不追求效果：
它是一张"训练链路体检表"，不是可用模型。

## 前置

075、077（会推理，也要会拼窗口）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `one_step_train.py` | MONAI 变换链 + 缩小 UNet + DiceCE + Novograd，跑一步并逐项断言 | 博文代码块 #3 |

## 运行

```bash
python -m pip install -r ../../requirements/advanced.txt

# 默认：合成数据、CPU、roi=32
python one_step_train.py

# 用真实的一对 NIfTI（需要 mm 单位的三维体积）
python one_step_train.py --image case_ct.nii.gz --label case_seg.nii.gz --device cuda:0 --roi 32
```

参数：`--image` / `--label`（必须同时给）、`--device {cpu,cuda:0}`、`--roi`（默认 32，
须是不小于 8 的 4 的倍数）。

## 环境要求

- `torch` + `monai`（`requirements/advanced.txt`）；读真实 NIfTI 还需 `nibabel`
- CPU 可跑（本机实测约 6 秒）；`cuda:0` 需 N 卡
- `static_check.py` 会标成 `需 GPU`：`--device` 的可选值里含字符串 `cuda:0`，
  触发了保守的文本匹配，属误报——默认路径是 CPU

## 观察点

- 最大参数变化 ≈ 学习率（0.002）：这说明优化器的归一化按预期工作。
- loss 0.57 只说明链路通，不代表模型学到了什么。

## 验证状态

**已实测通过**（CPU + 合成数据，2026-10-28，torch 2.13.0 / monai 1.6.1）。
输出见 `expected_output.md`。

真实 NIfTI 路径（`--image/--label`）未验证。

## 出处

- 博文：[78-器官已经标出来了AI还要学什么.md](../../../blog/078-器官已经标出来了AI还要学什么.md)
- 来源代码块 #3，126 行，md5 `c9c6127836b6ce42ef21e3bec44b1608`

代码逐字一致。**博文未指定文件名**，`one_step_train.py` 为仓库新增命名。
