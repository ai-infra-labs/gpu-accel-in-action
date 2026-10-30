# 010 · PyCUDA 全局归约与前缀和

GPU 上把 N 个数变成一个数（Reduction），或者把 N 个数变成 N 个前缀结果（Scan），
难点不在加法本身，而在**局部信息如何跨 Block 汇成全局信息**——`__syncthreads()`
管不了普通 Grid，只能靠多级 Kernel 启动与中间缓冲区接力。

本篇用两段式实现：Block 内折半树（归约）/ 倍增（Scan），再靠递归 workspace
逐级合并，最后回填偏移。

## 前置

001（线程块与网格）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `reduction_scan.py` | 三个 kernel + 启动路径 + 最小 `main()` | 博文 §6.3 代码块 |
| `verify_reduction_scan.py` | 正确性套件 + Benchmark 工具 | 博文附录 A 代码块 |

两个文件必须放在同一目录：后者 `from reduction_scan import ...`。

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt

python reduction_scan.py                        # 最小入口：跑一次 Reduction 与 Scan
python verify_reduction_scan.py --test quick    # 小规模，先确认环境可用
python verify_reduction_scan.py --test full     # 完整回归 + benchmark
```

`--test quick` 是环境探针，不是正确性证明；只有 `full` 覆盖递归 scan 路径。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit（PyCUDA 编译内核依赖 `nvcc`，只装 pip 包不够）
- 依赖层：`requirements/python-gpu.txt`
- `block_size` 必须是 2 的幂：归约用折半树、Scan 用倍增步长，两者都依赖这一点

## 观察点

- 归约轮数是 log2(n)；与串行求和的差异应在 1e-6 量级（float32 累加顺序不同）。
- 前缀和的验证脚本会跑多种规模，任一规模失败都说明同步点不对。

## 验证状态

**未实测。** 本机为 Apple Silicon，无 `nvcc`、无 NVIDIA GPU，仅完成：

- [x] 语法检查（`scripts/static_check.py`）
- [ ] 编译并运行 `reduction_scan.py`
- [ ] `--test quick` 通过
- [ ] `--test full` 通过
- [ ] 性能数据采集

实测输出与踩坑记录写入 `expected_output.md`。

## 出处

- 博文：[10-pycuda-reduction-prefix-sum-light-tech.md](../../../blog/010-PyCUDA归约与前缀和.md)
- `reduction_scan.py` ← 该文代码块 #22（281 行，md5 `00c4c1604417bfed750826243d68aa44`）
- `verify_reduction_scan.py` ← 该文代码块 #37（435 行，md5 `cab190d074782d209cb8c012aee4231f`）

代码与博文逐字一致，未做改写。若在仓库中修改，必须同步回博文，否则读者
复制博文代码会与仓库对不上。
