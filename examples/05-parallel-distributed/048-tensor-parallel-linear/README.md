# 048 · 一层矩阵太大，怎样让多张 GPU 合力算？

张量并行（TP）把一层线性层按列切成两半：每张卡只算一半的输出维度，
最后用一次 AllReduce 把各自的部分和加起来。本篇用一个 1×4 的输入把
"切开—各算—求和"三步算清楚。

## 前置

047（集合通信原语）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `tp_linear_demo.py` | 两层 TP 切分 + AllReduce 前向，与整模型结果逐值对齐 | 博文代码块 #4，需 2 卡 |
| `tp_split_check.py` | 纯 Python 版切分推演：按输出切 / 按输入切 / 两层串接 / B=2 | 博文代码块 #1，CPU 可跑 |

## 运行

```bash
# 不需要 GPU：纯 Python 复现切分与部分和（几秒出结果）
python tp_split_check.py

# 需要 2 张卡：真正跑 NCCL 的 TP 前向
python -m pip install -r ../../requirements/advanced.txt
torchrun --nproc_per_node=2 tp_linear_demo.py
```

## 环境要求

- `tp_split_check.py`：无依赖（纯标准库）
- `tp_linear_demo.py`：2 张 NVIDIA GPU，NCCL 后端，`requirements/advanced.txt`

## 观察点

- 纯 Python 版打印 TP 两层结果: [12, 6]，与整模型一致；这就是多卡版的参照值。
- 偏置 bb 只能在 AllReduce 之后加一次；每个 rank 都加再求和会被乘上卡数。

## 验证状态

- `tp_split_check.py`：**已实测通过**（CPU，2026-10-28），输出见 `expected_output.md`
- `tp_linear_demo.py`：**未实测**，需 2 张 GPU 与 NCCL

两个脚本用的是同一组常量，纯 Python 版的结果 `[12, 6]` 就是 GPU 版应当打印的值——
这让"多卡跑出来的数对不对"有了一个不依赖显卡的参照。

## 出处

- 博文：[48-一层矩阵太大怎样让多张GPU合力算.md](../../../blog/048-一层矩阵太大怎样让多张GPU合力算.md)
- 代码块 #4，51 行，md5 `c69b50519290c91d920c06a4329c5178` → `tp_linear_demo.py`
- 代码块 #1，64 行，md5 `7122d31fb399a4f7edc3f8533677f8e5` → `tp_split_check.py`

代码逐字一致。**博文未指定文件名**，两个文件名为仓库新增。
