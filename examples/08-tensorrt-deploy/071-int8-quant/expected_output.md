# 071 · 实测输出

**状态：未实测（需 `nvidia-modelopt`）。**

## 1. 环境

```
平台 / Python：
nvidia-modelopt / onnx / onnxruntime / numpy：
校准集：calib_features.npy，形状 / dtype / 来源
输入模型：onnx_demo/small_classifier.onnx（sha256 待填）
```

## 2. 运行结果

| 项 | 值 |
|---|---|
| QuantizeLinear 数量 | 待填（必须 > 0） |
| DequantizeLinear 数量 | 待填（必须 > 0） |
| opset | 待填（脚本按 19 生成） |
| 输入/输出 elem_type | 应为 `FLOAT`（1） |
| `onnx.checker.check_model(full_check=True)` | 通过才会走到打印 |

判读方式：

- **Q/DQ 成对出现才有意义**。只有 Q 没有 DQ，说明后面的算子仍要 float，
  图却断了类型——加载时就会报类型不匹配。
- **Q/DQ 都集中在顶层图**：如果数量是 0，先看 `calibration_eps` 是否与你的
  环境匹配（本脚本写死 `["cpu"]`）。
- **输入输出必须还是 FP32**：`direct_io_types=False` 保留浮点边界，
  这样同一个推理调用代码不用改。

## 3. 这份数据不能说明什么

- **不能说明用了 INT8 指令**：Q/DQ 只是图上的类型标注，真正跑 INT8 需要后端
  支持（TensorRT 的 explicit quantization）。脚本最后一行也这么写。
- **不能说明精度损失可接受**：没有真实标注、没有评测，只有结构检查。
  用随机校准集跑通，只能证明流程能走完。
- **不代表速度**：没有计时。

## 4. 踩坑记录

待填。至少记录：

- 校准集必须是非空 `float32 [N, 256]`，含 NaN/Inf 会被 `FeatureCalibration` 直接拒绝
- `calibration_method="max"` 对异常值敏感：若校准集里有一个极大值，
  整个张量的 scale 会被拉粗，精度掉得比预期多
- 输出路径的父目录必须先存在，否则最后写文件时才会失败（前面的量化白跑）
