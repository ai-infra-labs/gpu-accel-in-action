"""权重删了一半，模型为什么没快？（第8章 · TensorRT 部署）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch、ONNX、ONNX Runtime、NumPy。
"""
import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


class SmallClassifier(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(256, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def compact_hidden(base, keep):
    # 本例只处理 CPU、FP32 的上述小网络；索引必须互异且有效。
    keep = torch.as_tensor(keep, dtype=torch.long)
    if (keep.ndim != 1 or keep.numel() == 0
            or keep.unique().numel() != keep.numel()
            or keep.min().item() < 0 or keep.max().item() >= 64):
        raise ValueError("keep 必须是 0～63 中非空、不重复的一维索引。")
    smaller = SmallClassifier(hidden=keep.numel()).eval()
    with torch.no_grad():
        smaller.fc1.weight.copy_(base.fc1.weight.index_select(0, keep))
        smaller.fc1.bias.copy_(base.fc1.bias.index_select(0, keep))
        smaller.fc2.weight.copy_(base.fc2.weight.index_select(1, keep))
        smaller.fc2.bias.copy_(base.fc2.bias)
    return smaller


def main():
    torch.manual_seed(42)
    base = SmallClassifier().cpu().eval()  # 随机模型，仅用于流程教学。
    keep = torch.arange(0, 64, 2)
    drop = torch.ones(64, dtype=torch.bool)
    drop[keep] = False

    # 大网络仍有 64 个隐藏单元，但被删除单元不再影响最终输出。
    masked_reference = copy.deepcopy(base)
    with torch.no_grad():
        masked_reference.fc2.weight[:, drop] = 0
    smaller = compact_hidden(base, keep)

    probes = [torch.zeros(1, 256), torch.randn(7, 256), torch.randn(3, 256) * 10]
    with torch.inference_mode():
        for x in probes:
            torch.testing.assert_close(
                smaller(x), masked_reference(x), atol=1e-5, rtol=1e-4
            )
    print("所测输入上，小网络与被屏蔽的大网络通过重建核对。")
    print("这不代表与原模型质量相同。")

    folder = Path("compact_demo")
    folder.mkdir(exist_ok=True)
    example = torch.randn(1, 256)  # 三份 ONNX 均固定 batch=1。
    for name, model in (("original", base),
                        ("masked", masked_reference), ("compact", smaller)):
        path = folder / f"{name}.onnx"
        torch.onnx.export(
            model, (example,), str(path), dynamo=True, opset_version=18,
            input_names=["features"], output_names=["logits"],
            external_data=False,
        )
        onnx.checker.check_model(str(path), full_check=True)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        # 每份导出产物与自己的源模型核对，避免比较错对象。
        with torch.inference_mode():
            expected = model(example).numpy()
        actual = session.run(["logits"], {"features": example.numpy()})[0]
        if actual.shape != expected.shape:
            raise AssertionError("导出前后输出形状不同。")
        np.testing.assert_allclose(
            actual, expected, atol=1e-5, rtol=1e-4, equal_nan=False
        )
        weights = sum(p.numel() for n, p in model.named_parameters()
                      if n.endswith("weight"))
        parameters = sum(p.numel() for p in model.parameters())
        print(name, "weight元素:", weights, "全部参数:", parameters)
    print("已导出模型；尚未构建或测量 TensorRT 引擎。")


if __name__ == "__main__":
    main()
