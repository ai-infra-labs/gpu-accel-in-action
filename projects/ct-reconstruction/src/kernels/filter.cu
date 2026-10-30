// filter.cu —— 一维斜坡滤波（简化版）
//
// 真实 FDK 在频域乘 |ω| 窗函数；这里用中心差分做出「去直流」这一个效果：
//     out[i] = 0.5·in[i] − 0.25·(in[i−1] + in[i+1])
// 系数和为 0，常量段过完是 0。定位见 README：它服务于把流程串起来，
// 不代表生产级重建质量。
//
// 分块方式：一个 block 负责一条投影，整行搬进 shared memory，
// 于是每个元素只从全局内存读一次，左右邻居都在片上。
#include "mi/kernels.cuh"

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 128;

__global__ void ramp_filter_kernel(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int nAngles, int nDet) {
    extern __shared__ float tile[];   // nDet + 2

    const int a = blockIdx.x;
    if (a >= nAngles) {
        return;
    }
    const float* src = in + static_cast<size_t>(a) * nDet;
    float* dst       = out + static_cast<size_t>(a) * nDet;

    for (int i = threadIdx.x; i < nDet; i += blockDim.x) {
        tile[i + 1] = src[i];         // 主体错一位放，两边留边界
    }
    if (threadIdx.x == 0) {
        tile[0]        = src[0];      // 左边界复制端点
        tile[nDet + 1] = src[nDet - 1];  // 右边界复制端点
    }
    // 没有这句，别的线程可能读到尚未写好的邻居
    __syncthreads();

    for (int i = threadIdx.x; i < nDet; i += blockDim.x) {
        dst[i] = 0.5f * tile[i + 1] - 0.25f * (tile[i] + tile[i + 2]);
    }
}

}  // namespace

void mi_ramp_filter1d(const float* d_in, float* d_out, int nAngles, int nDet) {
    const size_t sharedBytes = static_cast<size_t>(nDet + 2) * sizeof(float);
    ramp_filter_kernel<<<nAngles, kThreads, sharedBytes>>>(d_in, d_out, nAngles, nDet);
}
