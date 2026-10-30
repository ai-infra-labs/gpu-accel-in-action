# 022 · Loss 在下降，学生却没学到老师？一篇讲透知识蒸馏

学生模型直接学标签（CE）与学老师的软化输出（KD），差别不在损失数值大小，
而在**梯度里带了什么信息**：老师的 softmax 分布把"这几个类容易混"编码进去了。

本篇用 CIFAR-10 做三组对照（Teacher / 学生 CE / 学生 KD），并额外提供两条
不依赖数据集的梯度自检。

## 前置

020、021（先看量化，再看蒸馏；两者都在压缩模型）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `distill_cifar10.py` | 教师训练、CE 基线、KD 蒸馏、损失与网络级梯度自检 | 博文代码块 #12 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt

# 只跑梯度自检：不需要数据集，CPU 即可，几秒完成
python distill_cifar10.py --smoke-test

# 完整对照：需要 CIFAR-10（自动下载约 170 MB）与 GPU
python distill_cifar10.py --mode compare
```

常用参数：`--mode {teacher,ce,kd,compare}`、`--teacher-epochs 20`、
`--student-epochs 20`、`--temperature 2.0`、`--alpha 0.7`、`--device auto`。

## 环境要求

- `--smoke-test`：`torch` + `torchvision`，CPU 即可
- 完整对照：NVIDIA GPU（`--device cuda`）或足够的耐心（`--device cpu`，20 轮 × 2）
- 依赖层：`requirements/python-cpu.txt`（自检）／`advanced.txt`（完整对照）

`static_check.py` 会把它标成 `需 GPU*`——脚本里有 `torch.cuda.synchronize`，
属保守判定；`--smoke-test` 与 `--device cpu` 都不需要显卡。

## 观察点

- 看 CE 与 KD 的**差值**，不是绝对值：20 轮短训练里 KD 更差也完全可能。
- --smoke-test 的两条梯度自检必须先过；自检不过时，训练结果的任何差异都无法解释。

## 验证状态

**部分实测。**

- `--smoke-test`：**已实测通过**（CPU，2026-10-28），两条自检都打印"通过"
- `--mode compare`：**未实测**，需要 CIFAR-10 与训练时间

详见 `expected_output.md`。

## 出处

- 博文：[22-knowledge-distillation-cuda-to-llm.md](../../../blog/022-知识蒸馏：从CNN到大模型的思路与实践.md)
- 来源代码块 #12，355 行，md5 `8f615afcaaeab1aa1865ed7d032be690`

代码逐字一致。**博文未指定文件名**，`distill_cifar10.py` 为仓库新增命名。
