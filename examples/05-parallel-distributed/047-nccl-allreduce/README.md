# 047 · GPU 各算各的，结果怎么对齐？讲透 NCCL

两张卡各算一半梯度，最后要把两份加起来。AllReduce 做的是"每个 rank 拿到同一份
总和"；ReduceScatter + AllGather 则是"先各拿一段，再拼回来"。
本篇用 4 个数的极小例子把三条原语的语义钉死，不测带宽。

## 前置

046（为什么要对答案；046 只有片段）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `nccl_allreduce_torchrun.py` | AllReduce / AllGather / ReduceScatter 的语义校验 | 博文代码块 #9 |

## 运行

```bash
python -m pip install -r ../../requirements/advanced.txt
torchrun --nproc_per_node=2 nccl_allreduce_torchrun.py
```

必须用 `torchrun` 起两个进程，并且每张卡都要可见这两张 GPU。
直接 `python nccl_allreduce_torchrun.py` 会打印"请在单机两张 NVIDIA GPU 上"并退出。

## 环境要求

- 2 张 NVIDIA GPU（脚本检查 `device_count() >= 2`）
- `torch.distributed` 的 NCCL 后端
- 依赖层：`requirements/advanced.txt`（`mpi4py` 与本篇无关，同层装着而已）

## 观察点

- AllReduce 应得 [11,22,33,44]，且 ReduceScatter + AllGather 拼回同一结果。
- 三组 assert_close 全过才会打印四行；没有输出就是校验没过，不要只看有没有报错。

## 验证状态

**未实测。** 本机无 GPU，仅完成语法检查。

三组断言（AllReduce、AllGather、ReduceScatter 重组）全部通过才会打印结果，
所以"能打印出四行"本身就是通过证据。见 `expected_output.md`。

## 出处

- 博文：[47-GPU各算各的结果怎么对齐讲透NCCL.md](../../../blog/047-GPU各算各的结果怎么对齐讲透NCCL.md)
- 来源代码块 #9，62 行，md5 `c1a0b4dfa8f0b780f17b3bae08c1be9e`

代码逐字一致。**博文未指定文件名**，`nccl_allreduce_torchrun.py` 为仓库新增命名。
