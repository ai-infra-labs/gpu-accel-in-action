// ============================================================================
// v0_naive.cu —— 斜坡滤波（Ramp Filter）的最小教学版
//
// 只回答一个问题：一维卷积怎么在 GPU 上做，以及为什么需要 __syncthreads()。
//
// 定位：这是反投影的**前置**步骤，不是本篇主线。真实 FDK 的滤波在频域做
// （乘 |ω| 的窗函数），这里用中心差分把「去低频」这一个效果做出来：
//
//     out[i] = 0.5·in[i] − 0.25·(in[i−1] + in[i+1])
//
// 系数和为 0，所以常量输入（纯直流）过完是 0——这正是 ramp 滤波要做的事：
// 反投影会把低频反复叠加、越叠越糊，滤波先把直流和低频压掉。
// 它是 |ω| 滤波器的一阶近似，不是等价实现，别拿它当生产代码。
//
// 分工：一个 block 处理一条投影（一行 sinogram），整行搬进 shared memory，
// 中间两个邻居直接从 shared 读，于是每个元素只从全局内存读一次。
//
// 编译运行： nvcc -O3 -arch=native v0_naive.cu -o ramp_filter && ./ramp_filter
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define MI_CUDA_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "[MI] CUDA 错误 %s:%d -> %s\n", __FILE__,     \
                         __LINE__, cudaGetErrorString(err__));                 \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

constexpr int kNAngles = 180;   // 投影角度数
constexpr int kNDet    = 384;   // 每条投影的探测器像元数
constexpr int kBlock   = 128;   // 每个 block 的线程数

// 一个 block 一条投影行：shared 里存 nDet + 2，两端各留一个边界值
__global__ void ramp_filter_kernel(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int nAngles, int nDet) {
    extern __shared__ float tile[];          // 大小 nDet + 2

    const int a = blockIdx.x;                // 第 a 条投影
    if (a >= nAngles) {
        return;
    }
    const float* src = in + static_cast<size_t>(a) * nDet;
    float* dst       = out + static_cast<size_t>(a) * nDet;

    // 主体搬进 shared memory，tile[i+1] 对应 src[i]
    for (int i = threadIdx.x; i < nDet; i += blockDim.x) {
        tile[i + 1] = src[i];
    }
    // 边界：最左/最右各复制一份端点值，避免卷积越界
    if (threadIdx.x == 0) {
        tile[0]        = src[0];
        tile[nDet + 1] = src[nDet - 1];
    }
    // 没有这一句，别的线程可能读到还没写好的邻居——这是 shared memory 的经典坑
    __syncthreads();

    for (int i = threadIdx.x; i < nDet; i += blockDim.x) {
        const float left   = tile[i];
        const float center = tile[i + 1];
        const float right  = tile[i + 2];
        dst[i] = 0.5f * center - 0.25f * (left + right);
    }
}

// CPU 参考实现：用来给 kernel 对答案
static void ramp_filter_cpu(const std::vector<float>& in,
                            std::vector<float>& out,
                            int nAngles, int nDet) {
    out.assign(in.size(), 0.0f);
    for (int a = 0; a < nAngles; ++a) {
        const size_t base = static_cast<size_t>(a) * nDet;
        for (int i = 0; i < nDet; ++i) {
            const float l = in[base + (i > 0 ? i - 1 : i)];
            const float c = in[base + i];
            const float r = in[base + (i + 1 < nDet ? i + 1 : i)];
            out[base + i] = 0.5f * c - 0.25f * (l + r);
        }
    }
}

int main() {
    // 合成数据：一条随角度缓慢变化的投影，前半段是常量段（用于验证直流被压掉）
    std::vector<float> h_in(static_cast<size_t>(kNAngles) * kNDet);
    for (int a = 0; a < kNAngles; ++a) {
        for (int d = 0; d < kNDet; ++d) {
            const size_t idx = static_cast<size_t>(a) * kNDet + d;
            if (a < 8) {
                h_in[idx] = 1.0f;                       // 纯直流段
            } else {
                const float t = static_cast<float>(d) / kNDet;
                h_in[idx] = 1.0f + 0.5f * sinf(6.2831853f * t) *
                                   cosf(0.01f * static_cast<float>(a));
            }
        }
    }

    const size_t n = h_in.size();
    float* d_in  = nullptr;
    float* d_out = nullptr;
    MI_CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    MI_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    MI_CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                             cudaMemcpyHostToDevice));

    const size_t sharedBytes = (kNDet + 2) * sizeof(float);
    ramp_filter_kernel<<<kNAngles, kBlock, sharedBytes>>>(d_in, d_out, kNAngles, kNDet);
    MI_CUDA_CHECK(cudaGetLastError());
    MI_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(n);
    MI_CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float),
                             cudaMemcpyDeviceToHost));

    // 先校验，再谈性能
    std::vector<float> h_ref;
    ramp_filter_cpu(h_in, h_ref, kNAngles, kNDet);

    double maxAbs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        maxAbs = std::fmax(maxAbs, std::fabs(static_cast<double>(h_out[i]) - h_ref[i]));
    }

    // 直流段过完滤波器应接近 0：这是「系数和为 0」的直接证据
    double dcMax = 0.0;
    for (int a = 0; a < 8; ++a) {
        for (int d = 1; d < kNDet - 1; ++d) {
            dcMax = std::fmax(dcMax, std::fabs(static_cast<double>(
                                         h_out[static_cast<size_t>(a) * kNDet + d])));
        }
    }

    std::printf("[MI] 规模 nAngles=%d nDet=%d，block=%d，shared=%zu 字节\n",
                kNAngles, kNDet, kBlock, sharedBytes);
    std::printf("[MI] 与 CPU 参考最大绝对误差 %.3e\n", maxAbs);
    std::printf("[MI] 直流段滤波后最大幅值 %.3e（应接近 0）\n", dcMax);
    std::printf("[MI] 判定：%s（容差 1e-6）\n", maxAbs < 1e-6 ? "PASS" : "FAIL");

    MI_CUDA_CHECK(cudaFree(d_in));
    MI_CUDA_CHECK(cudaFree(d_out));
    return maxAbs < 1e-6 ? 0 : 1;
}
