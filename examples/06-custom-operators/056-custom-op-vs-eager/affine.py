"""编译并加载手写 CUDA affine+ReLU 算子，暴露 affine_relu 给基准脚本使用。

仅做算子构建：真正的「三种实现同场比较」在 bench_three_ways.py。
运行本文件要求本机有可用的 CUDA 工具链（nvcc）与 torch，否则会在
torch.cuda.is_available() 检查时报「需要可用的 CUDA GPU」并退出。
"""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

if not torch.cuda.is_available():
    raise RuntimeError("需要可用的 CUDA GPU")

root = Path(__file__).resolve().parent
load(
    name="affine_lab5_ext",
    sources=[str(root / "affine_op.cpp"), str(root / "affine_kernel.cu")],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2"],
    is_python_module=False,
    verbose=True,
)
affine_relu = torch.ops.affine_lab5.affine_relu.default
