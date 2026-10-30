# 056 · 手写 CUDA 一定更快吗？三种实现同场比较

同一道 `relu(x * 2 - 1)`，三种写法：PyTorch eager（两次 kernel）、
`torch.compile`（inductor 融合）、手写 CUDA 扩展（一次 kernel）。
本篇不比"谁最快"，而是把口径钉死——只有同一批数据、同一轮序、同一计时边界，
"快 1.2 倍"才有意义。

## 前置

053–055（自定义算子的最小流程、接入 PyTorch、梯度）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `affine_op.cpp` | 算子注册（`TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL`） | 块 #4，逐字 |
| `affine_kernel.cu` | CUDA kernel + 启动函数 | 块 #5 + #6，**拼接** |
| `affine.py` | 用 `cpp_extension.load` 编译并暴露 `affine_relu` | 块 #7，逐字 |
| `bench_three_ways.py` | 正确性核对 + 三种实现的计时与 profiler | 块 #8 + #9 + #10，**拼接** |

两个文件由博文的多个代码块按顺序拼接而成（单块缺一半，无法独立运行）。
拼接规则与登记见 [`docs/PROVENANCE.json`](../../../docs/maintaining/provenance.json)，
用 `python3 scripts/build_from_blocks.py --check` 可以验证拼接结果与博文一致。

## 运行

```bash
python -m pip install -r ../../requirements/advanced.txt
cd 056-custom-op-vs-eager
python bench_three_ways.py            # 首次会编译扩展，耗时较长
PROFILE=1 python bench_three_ways.py  # 额外导出 Chrome trace
```

产物：`affine_relu_trace.json`（仅 `PROFILE=1` 时）、`torch_extensions/` 下的构建缓存。

## 环境要求

- NVIDIA GPU + CUDA Toolkit（`nvcc` 必须可用，`cpp_extension.load` 要现场编译）
- 依赖层：`requirements/advanced.txt`

## 观察点

- 小 N（1024）三者打平：时间几乎全是启动开销，别在这里下结论。
- 大 N 上看两点：fused 相对 eager 的收益（两次 kernel 变一次）、compiled 是否可用。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查。

需要记录：`N=1_024 / 65_536 / 1_000_000 / 8_000_000` 四组下三种实现的
median/p25/p75 与 `eager/time` 比值；`compiled` 是否可用。
见 `expected_output.md`。

## 出处

- 博文：[56-手写CUDA一定更快吗三种实现同场比较.md](../../../blog/056-手写CUDA vs PyTorch vs cuBLAS性能对比.md)
- 块 #4 / #7 逐字入库；`affine_kernel.cu` = 块 #5 + #6；`bench_three_ways.py` = 块 #8 + #9 + #10

文件名沿用块 #7 里 `sources=["affine_op.cpp", "affine_kernel.cu"]` 的写法。
