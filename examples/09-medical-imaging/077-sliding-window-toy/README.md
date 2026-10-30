# 077 · CT 放不进显存，切成小块后怎样拼回去

整份三维 CT 塞不进显存，就切成 `96×96×96` 的窗口一块块推理。切不难，难的是
拼回去：两块窗口覆盖同一位置时可能给出不同分数，最后该听谁的？

本篇只做**算法层的窗口与融合**，不训练网络，也不涉及真实数据。

## 前置

075（单窗口推理已跑通）。

## 文件

| 文件 | 作用 | 运行环境 |
|---|---|---|
| `sliding_window_toy.py` | NumPy 教学版：窗口起点、覆盖率、高斯权重、三窗投票融合 | 纯 CPU，仅需 numpy |
| `monai_window_check.py` | MONAI `sliding_window_inference` 的张量/设备连接检查（假预测器） | 需 torch + monai |

## 运行

```bash
# 纯 NumPy，几秒出结果，含 32 组断言
python sliding_window_toy.py

# MONAI 连接检查：默认 CPU
python -m pip install -r ../../requirements/advanced.txt
python monai_window_check.py            # 或 --device cuda:0
```

`monai_window_check.py` 里的 `Pointwise` 是**假预测器**（输出 `[-x, x]`），
它只用来证明窗口确实进了模型所在设备，不产生任何有意义的器官分割结果。

## 环境要求

- `sliding_window_toy.py`：`numpy`（`requirements/python-cpu.txt`）
- `monai_window_check.py`：`torch` + `monai`（`requirements/advanced.txt`）；
  `--device cuda:0` 需 N 卡

`static_check.py` 会把 `monai_window_check.py` 标成「需 GPU」——它的 `--device`
可选值里含字符串 `cuda:0`，触发了保守的文本匹配，属误报。默认的 CPU 路径
不需要显卡。

## 观察点

- 窗口数随 overlap 变化：0/0.25/0.5/0.75 对应 27/64/125/512。
- 重叠区的权重和必须是 1；归一化漏了会让边界变亮或变暗。

## 验证状态

- `sliding_window_toy.py`：**已实测通过**（CPU，2026-10-28）
- `monai_window_check.py`：**未实测**（本机未装 torch / monai，仅语法检查）

## 出处

- 博文：[77-CT放不进显存切成小块后怎样拼回去.md](../../../blog/077-CT放不进显存切成小块后怎样拼回去.md)
- 代码块 #0，125 行，md5 `2c6299f2e1c14499364bac880100c978` → `sliding_window_toy.py`
- 代码块 #4，46 行，md5 `37c130951bf89ff8e3b4cc3efc7e66c6` → `monai_window_check.py`

代码逐字一致。**博文未指定文件名**，两个文件名为仓库新增。
