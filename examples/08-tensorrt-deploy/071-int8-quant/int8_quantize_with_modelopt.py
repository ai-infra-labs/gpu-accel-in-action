"""模型变成 INT8，精度到底丢在哪里？（第8章 · TensorRT 部署）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：ONNX、ONNX Runtime、NumPy。
"""
from collections import Counter
from importlib.metadata import version

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader
from modelopt.onnx.quantization import quantize


class FeatureCalibration(CalibrationDataReader):
    def __init__(self, path):
        self.rows = np.load(path, allow_pickle=False)
        if (self.rows.dtype != np.float32 or self.rows.ndim != 2
                or self.rows.shape[1] != 256 or len(self.rows) == 0):
            raise ValueError("校准数据必须是非空 float32 [N,256] 数组。")
        if not np.isfinite(self.rows).all():
            raise ValueError("校准数据包含 NaN 或 Inf。")
        self.rewind()

    def get_next(self):
        if self.index >= len(self.rows):
            return None
        row = self.rows[self.index:self.index + 1]
        self.index += 1
        return {"features": np.ascontiguousarray(row)}

    def rewind(self):
        self.index = 0


def main():
    for name in ("nvidia-modelopt", "onnx", "onnxruntime", "numpy"):
        print(name, version(name))
    output_path = "onnx_demo/small_classifier.int8.onnx"
    quantize(
        onnx_path="onnx_demo/small_classifier.onnx",
        quantize_mode="int8",
        calibration_data_reader=FeatureCalibration("calib_features.npy"),
        calibration_method="max",
        calibration_eps=["cpu"],
        high_precision_dtype="fp32",
        direct_io_types=False,
        opset=19,
        output_path=output_path,
    )
    onnx.checker.check_model(output_path, full_check=True)
    model = onnx.load(output_path)
    counts = Counter(node.op_type for node in model.graph.node)
    print("顶层图的 Q/DQ 数量:", counts["QuantizeLinear"], counts["DequantizeLinear"])
    if counts["QuantizeLinear"] == 0 or counts["DequantizeLinear"] == 0:
        raise RuntimeError("本例未得到预期 Q/DQ，请检查量化覆盖与日志。")
    for tensor in list(model.graph.input) + list(model.graph.output):
        if tensor.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise RuntimeError("本例运行器要求 FP32 输入输出，请核对转换结果。")
    print("opset:", [(item.domain, item.version) for item in model.opset_import])
    print("已生成量化图；尚未证明 GPU 使用 INT8 或任务质量达标。")


if __name__ == "__main__":
    main()
