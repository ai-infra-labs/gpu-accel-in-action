# 071 · 模型变成 INT8，精度到底丢在哪里？

INT8 量化不是"把 float 换成 int8"这一句话：要定标定（scale）、要选校准集、
要在图里插 Q/DQ 节点、还要决定哪些算子保持高精度。本篇用
`nvidia-modelopt` 对一个现成 ONNX 做 INT8 量化，然后**只检查图结构**，
不宣称任何速度与精度收益。

## 前置

067（需要导出的 ONNX）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `int8_quantize_with_modelopt.py` | 校准集读取 + INT8 量化 + Q/DQ 与输入输出类型核对 | 博文代码块 #2 |

## 运行

前置两份输入，都放在当前目录：

```
onnx_demo/small_classifier.onnx    # 067 导出
calib_features.npy                 # float32 [N, 256]，校准集
```

```bash
python -m pip install -r ../../requirements/advanced.txt
cd 071-int8-quant
python ../067-onnx-export-alignment/export_and_check.py     # 生成 ./onnx_demo/*.onnx
# 合成一份校准集（仅用于跑通流程，真实量化必须用真实数据）
python -c "import numpy as np; np.save('calib_features.npy', np.random.default_rng(0).standard_normal((64, 256)).astype(np.float32))"
python int8_quantize_with_modelopt.py
```

产物：`onnx_demo/small_classifier.int8.onnx`。

## 环境要求

- `nvidia-modelopt`（`import modelopt`）、`onnx`、`onnxruntime`
- 量化在校准阶段用 CPU（`calibration_eps=["cpu"]`），不需要 GPU
- 依赖层：`requirements/advanced.txt`

## 观察点

- Q/DQ 必须成对且都大于 0；只有 Q 没有 DQ，图会类型不连续。
- 输入输出仍应是 FP32：direct_io_types=False 的意义就在这里。

## 验证状态

**未实测。** 本机未装 `nvidia-modelopt`，仅完成语法检查。

需要记录：各包版本、Q/DQ 节点数量、opset、输入输出是否仍为 FP32。
见 `expected_output.md`。

## 出处

- 博文：[71-模型变成INT8精度到底丢在哪里.md](../../../blog/071-模型变成INT8精度到底丢在哪里.md)
- 来源代码块 #2，60 行，md5 `9f4c0218701ceaacdba74d8abf116575`

代码逐字一致。**博文未指定文件名**，`int8_quantize_with_modelopt.py` 为仓库新增命名。
