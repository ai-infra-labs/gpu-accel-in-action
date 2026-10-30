# 069 · 引擎已经生成，为什么推理还会报错？

引擎构建成功 ≠ 能跑。常见三类错都发生在**运行时**：形状没落实（动态引擎）、
输出缓冲区没查询、地址没重新绑定。本篇把"加载—设形状—绑地址—取输出"整条路径
写全，并用不同 batch 反复切换来暴露"只在第一次正确"的写法。

## 前置

068（必须先有 engine）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `run_engine.py` | 加载 engine → 按 batch 落实形状 → 与 ONNX Runtime 逐值对齐 | 博文代码块 #2 |

## 运行

前置：`onnx_demo/small_classifier.engine` 与同名 `.onnx`（068 构建引擎、067 导出模型）。
两者要放在同一目录，脚本用 `path.with_suffix(".onnx")` 找参考模型。

```bash
python -m pip install -r ../../requirements/advanced.txt
python run_engine.py                      # 默认 onnx_demo/small_classifier.engine
python run_engine.py /path/to/x.engine    # 或显式指定
```

## 环境要求

- NVIDIA GPU + **TensorRT 11.x**（脚本检查主版本）
- `onnxruntime` 作为 CPU 参考实现
- 依赖层：`requirements/advanced.txt`

## 观察点

- 动态引擎要跑 8→1 的 batch 切换：只在第一次落实形状的写法会在这里崩。
- 误差 1e-6 量级；1e-1 先怀疑参考与引擎喂的不是同一份数据。

## 验证状态

**未实测。** 本机无 TensorRT，仅完成语法检查。

需要记录：动态/静态引擎的判定结果、各 batch × scale 的最大绝对误差。
见 `expected_output.md`。

## 被 070 复用

`08-tensorrt-deploy/070-app-latency/app_latency_bench.py` 里的 `from run_engine import FeatureRunner`
指的就是本文件。跑 070 时把它加进 `PYTHONPATH`：

```bash
cd 070-app-latency
PYTHONPATH=../069-engine-runtime python app_latency_bench.py
```

## 出处

- 博文：[69-引擎已经生成为什么推理还会报错.md](../../../blog/069-引擎已经生成为什么推理还会报错.md)
- 来源代码块 #2，125 行，md5 `2d327f119e42d8c1cdf867400375bd79`

代码逐字一致。文件名沿用 070 的 `from run_engine import ...`。
