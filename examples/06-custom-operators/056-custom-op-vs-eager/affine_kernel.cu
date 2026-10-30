/*
 * 手写 CUDA 一定更快吗？三种实现同场比较（第6章 · 自定义算子）。
 * 本文件是对应博文代码块的实现/演示；运行依赖：PyTorch（torch.utils.cpp_extension 编译）。
 */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

__global__ void affine_relu_kernel(const float* x, float* y, int64_t n,
                                    float scale, float bias) {
    const int64_t i = static_cast<int64_t>(blockIdx.x)
                      * blockDim.x + threadIdx.x;
    if (i < n) {
        // 分别按 float32 舍入乘法和加法，避免隐式合成 FMA。
        const float value = __fadd_rn(__fmul_rn(x[i], scale), bias);
        y[i] = value > 0.0f ? value : 0.0f;
    }
}

at::Tensor affine_relu_cuda(const at::Tensor& x, double scale, double bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.layout() == c10::kStrided && x.is_contiguous(),
                "x must have contiguous strided layout");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(!x.requires_grad(), "this forward-only demo has no autograd");
    TORCH_CHECK(x.numel() <= std::numeric_limits<int>::max(),
                "too many elements for this tutorial");

    const c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty_like(x);
    const int64_t n = x.numel();
    if (n == 0) return y;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    affine_relu_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n,
        static_cast<float>(scale), static_cast<float>(bias));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
