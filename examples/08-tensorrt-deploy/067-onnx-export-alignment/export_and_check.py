"""模型导出成功，为什么结果却变了？（第8章 · TensorRT 部署）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch、ONNX、ONNX Runtime、NumPy。
"""
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


class SmallClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # 输入 [1, 256]，隐藏特征 [1, 64]，输出 logits [1, 10]。
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    # 打印实际环境；后续复现实验时应保留这份记录。
    for package in ("torch", "numpy", "onnx", "onnxscript", "onnxruntime"):
        print(f"{package}={version(package)}")

    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    model = SmallClassifier().cpu().eval()
    # 这是随机初始化模型，用于验证导出流程，不用于评价分类能力。

    # 单个样例提供输入形状；本篇固定 batch=1，不声明动态维度。
    example = torch.from_numpy(rng.standard_normal((1, 256)).astype(np.float32))
    output_dir = Path("onnx_demo")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "small_classifier.onnx"

    torch.onnx.export(
        model,
        (example,),
        str(model_path),
        input_names=["features"],
        output_names=["logits"],
        dynamo=True,
        opset_version=18,
        external_data=False,
        report=True,
        artifacts_dir=str(output_dir),
    )
    # opset 18 是本例的明确选择，接收端仍需支持它和实际导出的算子。
    onnx.checker.check_model(str(model_path), full_check=True)

    # 保存 PyTorch 权重；下一篇直接加载同一份权重核对 TensorRT，
    # 不依赖相同随机种子在不同环境重新生成同一模型。
    torch.save(model.state_dict(), output_dir / "small_classifier.pt")

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    in_meta = session.get_inputs()[0]
    out_meta = session.get_outputs()[0]
    print("ONNX 输入:", in_meta.name, in_meta.shape, in_meta.type)
    print("ONNX 输出:", out_meta.name, out_meta.shape, out_meta.type)
    if in_meta.type != "tensor(float)" or out_meta.type != "tensor(float)":
        raise AssertionError(
            f"输入输出类型与约定不符: {in_meta.type} / {out_meta.type}"
        )

    # 多测几种输入；仍然不是覆盖所有输入的数学证明。
    cases = [("zeros", np.zeros((1, 256), dtype=np.float32))]
    for scale in (0.1, 1.0, 10.0):
        for index in range(4):
            x = (rng.standard_normal((1, 256)) * scale).astype(np.float32)
            cases.append((f"scale={scale}, case={index}", x))

    # 仅是本例浮点结果比较的起始容限，不是所有模型的统一标准。
    atol, rtol = 1e-5, 1e-4
    worst_abs_error = 0.0
    references = []
    for name, x in cases:
        with torch.inference_mode():
            reference = model(torch.from_numpy(x)).numpy()
        actual = session.run(["logits"], {"features": x})[0]
        references.append(reference)

        # 先核对形状，避免数组广播掩盖接口错误。
        if actual.shape != reference.shape:
            raise AssertionError(f"{name}: 输出形状不同")
        if not (np.isfinite(actual).all() and np.isfinite(reference).all()):
            raise AssertionError(f"{name}: 输出包含 NaN 或 Inf")
        # 容差比较通过不代表 dtype 相同，这里显式核对约定的 float32。
        if actual.dtype != np.float32:
            raise AssertionError(f"{name}: 输出 dtype={actual.dtype}，约定为 float32")

        error = float(np.max(np.abs(actual - reference)))
        worst_abs_error = max(worst_abs_error, error)
        if not np.allclose(actual, reference, atol=atol, rtol=rtol):
            diff = np.abs(actual - reference)
            pos = np.unravel_index(np.argmax(diff), diff.shape)
            raise AssertionError(
                f"{name}: 超出容限，最大绝对误差={error:.3e}，"
                f"位置={pos}，参考值={reference[pos]:.6e}，实际值={actual[pos]:.6e}"
            )

    # 保存本批输入与 PyTorch 参考输出、容差和依赖版本，
    # 下一篇核对 TensorRT 时直接复用，不再重新随机生成样本。
    np.savez(
        output_dir / "reference_inputs_outputs.npz",
        atol=np.array(atol),
        rtol=np.array(rtol),
        **{f"x_{i}": x for i, (_, x) in enumerate(cases)},
        **{f"reference_{i}": r for i, r in enumerate(references)},
    )

    print(f"通过 {len(cases)} 组数值核对；最大绝对误差={worst_abs_error:.3e}")
    print("已保存权重与参考数据:", output_dir / "small_classifier.pt")
    print("已保存输入与参考输出:", output_dir / "reference_inputs_outputs.npz")
    print("这不代表模型分类准确，也不代表 TensorRT 已通过验证。")


if __name__ == "__main__":
    main()
