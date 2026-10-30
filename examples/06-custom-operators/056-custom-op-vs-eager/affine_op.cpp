/*
 * 手写 CUDA 一定更快吗？三种实现同场比较（第6章 · 自定义算子）。
 * 本文件是对应博文代码块的实现/演示；运行依赖：PyTorch（torch.utils.cpp_extension 编译）。
 */
#include <ATen/ATen.h>
#include <torch/library.h>

at::Tensor affine_relu_cuda(const at::Tensor& x, double scale, double bias);

TORCH_LIBRARY(affine_lab5, m) {
    m.def("affine_relu(Tensor x, float scale, float bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(affine_lab5, CUDA, m) {
    m.impl("affine_relu", &affine_relu_cuda);
}
