// ============================================================================
// v1_optimized.cu —— 反投影：先改线程映射与访存，再谈别的
//
// 相对 v0 的四处改动，全部围绕「访存」而不是「算法」：
//   1) 线程映射从「一维摊平」改成 block(32,8) 二维网格
//      —— 一个 warp 的 32 个线程落同一行、相邻 32 列，探测器索引也相邻，
//         于是同一角度下 32 次取样压成少量内存事务；
//         同时省掉 idx 的整数除法和取模。
//   2) sinogram 行首指针逐角度递进，替掉每轮的 size_t 乘法。
//   3) 取样走 __ldg（只读数据缓存），sinogram 全程只读，用上纹理路径。
//   4) 双线性从两乘两加改写成 s0 + (s1 - s0) * f，少一次乘法。
//
// 编译运行： nvcc -O3 -arch=native v1_optimized.cu -o v1_optimized && ./v1_optimized
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#ifndef MI_CUDA_CHECK
#define MI_CUDA_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "[MI] CUDA 错误 %s:%d -> %s\n", __FILE__,     \
                         __LINE__, cudaGetErrorString(err__));                 \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

#ifndef MI_GEOM_DEFINED
#define MI_GEOM_DEFINED
struct MiGeom {
    int   nx;
    int   ny;
    int   nDet;
    int   nAngles;
    float halfExtent;
    float detSpacing;
    float dTheta;
};

inline MiGeom mi_default_geom() {
    MiGeom g;
    g.nx = 256;
    g.ny = 256;
    g.nDet = 384;
    g.nAngles = 180;
    g.halfExtent = 1.0f;
    g.detSpacing = 2.6f / static_cast<float>(g.nDet);
    g.dTheta = 3.14159265358979323846f / static_cast<float>(g.nAngles);
    return g;
}
#endif

#ifndef MI_SHARED_HOST
#define MI_SHARED_HOST
struct MiEllipse {
    float cx, cy, a, b, rho;
};

inline const MiEllipse* mi_phantom(int& count) {
    static const MiEllipse kEllipses[] = {
        { 0.00f,  0.00f, 0.72f, 0.90f,  1.00f},
        { 0.00f,  0.00f, 0.60f, 0.78f, -0.35f},
        { 0.30f,  0.10f, 0.22f, 0.30f,  0.45f},
        {-0.28f, -0.12f, 0.18f, 0.24f,  0.35f},
        { 0.02f, -0.45f, 0.12f, 0.10f,  0.60f},
    };
    count = static_cast<int>(sizeof(kEllipses) / sizeof(kEllipses[0]));
    return kEllipses;
}

inline void mi_build_sinogram(std::vector<float>& sino, const MiGeom& g) {
    sino.assign(static_cast<size_t>(g.nAngles) * g.nDet, 0.0f);
    int nEll = 0;
    const MiEllipse* ell = mi_phantom(nEll);
    const float uCenter = 0.5f * static_cast<float>(g.nDet - 1);

    for (int a = 0; a < g.nAngles; ++a) {
        const float theta = static_cast<float>(a) * g.dTheta;
        const float c = cosf(theta);
        const float s = sinf(theta);
        for (int d = 0; d < g.nDet; ++d) {
            const float t = (static_cast<float>(d) - uCenter) * g.detSpacing;
            float acc = 0.0f;
            for (int e = 0; e < nEll; ++e) {
                const MiEllipse& el = ell[e];
                const float tc   = t - (el.cx * c + el.cy * s);
                const float rad2 = el.a * el.a * c * c + el.b * el.b * s * s;
                const float rad  = sqrtf(rad2);
                if (fabsf(tc) < rad) {
                    acc += el.rho * (2.0f * el.a * el.b / rad2) *
                           sqrtf(rad2 - tc * tc);
                }
            }
            sino[static_cast<size_t>(a) * g.nDet + d] = acc;
        }
    }
}
#endif

#define MI_FLOP_PER_SAMPLE 14

// 双线性取样：__ldg 让编译器走只读数据缓存，且改写为「基准 + 增量」
__device__ __forceinline__ float bp_sample_v1(const float* __restrict__ row,
                                              float u, int nDet) {
    if (u <= 0.0f) {
        return __ldg(row);
    }
    const float last = static_cast<float>(nDet - 1);
    if (u >= last) {
        return __ldg(row + (nDet - 1));
    }
    const int   i0 = static_cast<int>(u);
    const float f  = u - static_cast<float>(i0);
    const float s0 = __ldg(row + i0);
    const float s1 = __ldg(row + i0 + 1);
    return s0 + (s1 - s0) * f;   // 一次乘法，等价于两端加权
}

__global__ void bp_v1_kernel(const float* __restrict__ sino,
                             float* __restrict__ vol,
                             MiGeom g) {
    // 二维线程映射：threadIdx.x 对应体素列，warp 内 32 个线程落在同一行
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= g.nx || iy >= g.ny) {
        return;
    }

    const float stepX = 2.0f * g.halfExtent / static_cast<float>(g.nx);
    const float stepY = 2.0f * g.halfExtent / static_cast<float>(g.ny);
    const float x = (static_cast<float>(ix) + 0.5f) * stepX - g.halfExtent;
    const float y = (static_cast<float>(iy) + 0.5f) * stepY - g.halfExtent;

    const float uCenter = 0.5f * static_cast<float>(g.nDet - 1);
    const float* row = sino;                 // 逐角度递进，不再每轮做乘法
    float sum = 0.0f;
    for (int a = 0; a < g.nAngles; ++a, row += g.nDet) {
        const float theta = static_cast<float>(a) * g.dTheta;
        const float p = x * cosf(theta) + y * sinf(theta);
        const float u = p / g.detSpacing + uCenter;
        sum += bp_sample_v1(row, u, g.nDet);
    }
    // 二维索引直接算出一维下标，省掉除法和取模
    vol[iy * g.nx + ix] = sum * g.dTheta;
}

void mi_launch_v1(const float* d_sino, float* d_vol, MiGeom g) {
    const dim3 block(32, 8);     // 一个 warp = 同一行相邻 32 列
    const dim3 grid((g.nx + block.x - 1) / block.x,
                    (g.ny + block.y - 1) / block.y);
    bp_v1_kernel<<<grid, block>>>(d_sino, d_vol, g);
}

#ifndef MI_NO_MAIN
int main() {
    const MiGeom g = mi_default_geom();

    std::vector<float> h_sino;
    mi_build_sinogram(h_sino, g);

    const size_t nVox  = static_cast<size_t>(g.nx) * g.ny;
    const size_t nSino = h_sino.size();

    float* d_sino = nullptr;
    float* d_vol  = nullptr;
    MI_CUDA_CHECK(cudaMalloc(&d_sino, nSino * sizeof(float)));
    MI_CUDA_CHECK(cudaMalloc(&d_vol, nVox * sizeof(float)));
    MI_CUDA_CHECK(cudaMemcpy(d_sino, h_sino.data(), nSino * sizeof(float),
                             cudaMemcpyHostToDevice));

    for (int i = 0; i < 5; ++i) {
        mi_launch_v1(d_sino, d_vol, g);
    }
    MI_CUDA_CHECK(cudaDeviceSynchronize());

    const int   kIters = 50;
    cudaEvent_t start, stop;
    MI_CUDA_CHECK(cudaEventCreate(&start));
    MI_CUDA_CHECK(cudaEventCreate(&stop));
    MI_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIters; ++i) {
        mi_launch_v1(d_sino, d_vol, g);
    }
    MI_CUDA_CHECK(cudaEventRecord(stop));
    MI_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    MI_CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    const double ms = elapsedMs / kIters;

    std::vector<float> h_vol(nVox);
    MI_CUDA_CHECK(cudaMemcpy(h_vol.data(), d_vol, nVox * sizeof(float),
                             cudaMemcpyDeviceToHost));

    double mean = 0.0;
    for (size_t i = 0; i < nVox; ++i) {
        mean += h_vol[i];
    }
    mean /= static_cast<double>(nVox);

    const double bytes = static_cast<double>(nVox) * g.nAngles * 2.0 * sizeof(float);
    const double flops = static_cast<double>(nVox) * g.nAngles * MI_FLOP_PER_SAMPLE;

    std::printf("[MI] 规模 nx=%d ny=%d nDet=%d nAngles=%d\n",
                g.nx, g.ny, g.nDet, g.nAngles);
    std::printf("[MI] v1 时间 %.3f ms（%d 次平均，5 次预热）\n", ms, kIters);
    std::printf("[MI] v1 有效带宽 %.1f GB/s（按 nx*ny*nAngles*8 字节计）\n",
                bytes / (ms * 1e6));
    std::printf("[MI] v1 估算算力 %.1f GFLOP/s（按每样本 %d FLOP 计）\n",
                flops / (ms * 1e6), MI_FLOP_PER_SAMPLE);
    std::printf("[MI] v1 中心体素 %.6f，全图均值 %.6f\n",
                h_vol[(g.ny / 2) * g.nx + g.nx / 2], mean);

    MI_CUDA_CHECK(cudaEventDestroy(start));
    MI_CUDA_CHECK(cudaEventDestroy(stop));
    MI_CUDA_CHECK(cudaFree(d_sino));
    MI_CUDA_CHECK(cudaFree(d_vol));
    return 0;
}
#endif
