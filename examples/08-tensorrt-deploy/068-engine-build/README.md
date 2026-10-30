# 068 · TensorRT 没改模型，为什么还能更快？

同一份 ONNX，交到 TensorRT 手里会被重排：算子融合、kernel 自动选择、
按形状特化。本篇只做「构建 + 信息导出」，不跑推理——先把"引擎里到底有什么"
看清楚，再谈快慢。

## 前置

067（先有 ONNX，才能建引擎）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `build_engine_and_report.py` | 解析 ONNX → 构建 engine → 导出 engine 信息 JSON | 博文代码块 #3 |

## 运行

前置：需要一份 `onnx_demo/small_classifier.onnx`，由 067 的脚本导出。两个脚本都按
**当前工作目录**写 `onnx_demo/`，所以要在本目录里跑 067 的脚本，产物才会落在
本脚本查找的位置：

```bash
python -m pip install -r ../../requirements/advanced.txt
python ../067-onnx-export-alignment/export_and_check.py   # 生成 ./onnx_demo/small_classifier.onnx
python build_engine_and_report.py
```

产物：`onnx_demo/small_classifier.engine`、`onnx_demo/small_classifier.engine.json`
（两者都在 `.gitignore` 里，不入库）。

## 环境要求

- NVIDIA GPU + **TensorRT 11.x**（脚本显式检查主版本号，10.x 会直接报错）
- 依赖层：`requirements/advanced.txt`

## 观察点

- 构建耗时是秒级，与推理耗时无关；别拿它和毫秒数比较。
- 引擎信息 JSON 里的融合层数，与 ONNX 节点数的差值就是「TensorRT 改了什么」。

## 验证状态

**未实测。** 本机无 TensorRT，仅完成语法检查。

脚本最后一行自己写着"输出正确性和性能仍未实测"——本篇不产生任何推理数字。
见 `expected_output.md`。

## 出处

- 博文：[68-TensorRT没改模型为什么还能更快.md](../../../blog/068-TensorRT没改模型为什么还能更快.md)
- 来源代码块 #3，65 行，md5 `47a1c2ff528acd213ead3e5185678bf2`

代码逐字一致。**博文未指定文件名**，`build_engine_and_report.py` 为仓库新增命名。
