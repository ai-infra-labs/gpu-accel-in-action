# 070 · 推理只要 1 毫秒，应用为什么还在等？

引擎执行 1 ms，端到端可能要 5 ms：差在 H2D 拷贝、D2H 拷贝、缓冲区分配与同步。
本篇用同一份引擎测三种口径——每次新建张量、复用缓冲区、纯引擎执行（CUDA 事件），
把"这 4 ms 去哪了"逐项摊开。

## 前置

069（运行时封装 run_engine 由本篇复用）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `app_latency_bench.py` | 三种口径的墙钟/事件计时 + 与 ONNX Runtime 的逐值核对 | 博文代码块 #2 |

## 运行

前置：`onnx_demo/small_classifier.engine` 与同名 `.onnx`（068/067 的产物），
以及 069 的 `run_engine.py`（本脚本 `from run_engine import FeatureRunner`）。

```bash
python -m pip install -r ../../requirements/advanced.txt
cd 070-app-latency
PYTHONPATH=../069-engine-runtime python app_latency_bench.py
PYTHONPATH=../069-engine-runtime python app_latency_bench.py onnx_demo/small_classifier.engine 8
```

参数：`engine 路径`（默认 `onnx_demo/small_classifier.engine`）、`batch`（默认 1）。

## 环境要求

- NVIDIA GPU + TensorRT 11.x + `onnxruntime`（CPU 参考）
- 必须先准备好 engine 与 `.onnx`，否则在构造 `FeatureRunner` 时就失败
- 依赖层：`requirements/advanced.txt`

## 观察点

- 三种口径的差值就是答案：分配开销、两次拷贝 + 同步、纯执行。
- 事件法数字最小且不含 H2D/D2H；它是「推理只要 1 ms」的那个 1 ms。

## 验证状态

**未实测。** 本机无 TensorRT，仅完成语法检查。

需要记录：两种墙钟口径的统计量、事件法平均值。见 `expected_output.md`。

## 出处

- 博文：[70-推理只要1毫秒应用为什么还在等.md](../../../blog/070-推理只要1毫秒应用为什么还在等.md)
- 来源代码块 #2，119 行，md5 `7963b52ea5dd43a6da717150bacfd553`

代码逐字一致。**博文未指定文件名**，`app_latency_bench.py` 为仓库新增命名。
