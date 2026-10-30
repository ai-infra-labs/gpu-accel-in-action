// ============================================================================
// v2_industrial.cu —— 反投影：把「每个体素每次都要算的东西」提到循环外
//
// 相对 v1 的改动只有一件事：预计算角度表，但它的收益和代价都值得看清。
//
//   改动 1：把 cos/sin 打包成 float2 放进常量内存（constant memory）
//           —— 常量内存走常量缓存，同一 warp 读同一地址时是「一次读取、全 warp
//              广播」，不像全局内存那样每个线程各发一次请求；
//           —— 更重要的是，每体素每角度的 2 次三角函数调用彻底消失。
//   改动 2：表里预先除好 detSpacing，内层循环少一次除法（除法比乘法贵得多）。
//   改动 3：__launch_bounds__(256) 告诉编译器 block 大小上限，让它敢多用寄存器；
//           配合 #pragma unroll 4 把角度循环摊开，掩盖常量内存与显存的延迟。
//
// 代价（这就是「工业里的权衡」）：
//   —— 常量内存只有 64 KB。float2 一条角度 8 字节，nAngles 上限 1024 条（8 KB），
//      超过就得回退到全局内存或改成分块，这里显式报错而不是静默降级；
//   —— 表按几何参数缓存，几何一变必须重传；规模越大收益越明显，小规模反而
//      可能因为多一次常量读取而持平。
//
// 编译运行： nvcc -O3 -arch=native v2_industrial.cu -o v2_industrial && ./v2_industrial
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

// 常量内存的容量上限：64 KB / 8 字节(float2) = 8192，这里保守取 1024 条角度，
// 留出余量给编译器可能生成的其他常量数据。
#define MI_MAX_ANGLES 1024

// 角度表：(cosθ / detSpacing, sinθ / detSpacing)
__constant__ float2 c_angle_cs[MI_MAX_ANGLES];

// 上传角度表：几何不变时只传一次
void mi_upload_angles_v2(MiGeom g) {
    static int   cachedAngles  = -1;
    static float cachedDTheta  = 0.0f;
    static float cachedSpacing = 0.0f;
    if (cachedAngles == g.nAngles && cachedDTheta == g.dTheta &&
        cachedSpacing == g.detSpacing) {
        return;
    }
    if (g.nAngles > MI_MAX_ANGLES) {
        std::fprintf(stderr,
                     "[MI] nAngles=%d 超过常量内存表上限 %d，请改用全局内存版本\n",
                     g.nAngles, MI_MAX_ANGLES);
        std::exit(EXIT_FAILURE);
    }

    std::vector<float2> table(static_cast<size_t>(g.nAngles));
    for (int a = 0; a < g.nAngles; ++a) {
        const float theta = static_cast<float>(a) * g.dTheta;
        table[a].x = cosf(theta) / g.detSpacing;
        table[a].y = sinf(theta) / g.detSpacing;
    }
    MI_CUDA_CHECK(cudaMemcpyToSymbol(c_angle_cs, table.data(),
                                     table.size() * sizeof(float2)));
    cachedAngles  = g.nAngles;
    cachedDTheta  = g.dTheta;
    cachedSpacing = g.detSpacing;
}

__device__ __forceinline__ float bp_sample_v2(const float* __restrict__ row,
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
    return s0 + (s1 - s0) * f;
}

__global__ void __launch_bounds__(256) bp_v2_kernel(
        const float* __restrict__ sino,
        float* __restrict__ vol,
        MiGeom g) {
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
    const float* row = sino;
    float sum = 0.0f;
#pragma unroll 4
    for (int a = 0; a < g.nAngles; ++a, row += g.nDet) {
        // 整个 warp 读同一条角度记录：常量缓存直接广播，无逐线程请求
        const float2 cs = c_angle_cs[a];
        const float  u  = x * cs.x + y * cs.y + uCenter;   // 无除法、无三角函数
        sum += bp_sample_v2(row, u, g.nDet);
    }
    vol[iy * g.nx + ix] = sum * g.dTheta;
}

void mi_launch_v2(const float* d_sino, float* d_vol, MiGeom g) {
    const dim3 block(32, 8);
    const dim3 grid((g.nx + block.x - 1) / block.x,
                    (g.ny + block.y - 1) / block.y);
    bp_v2_kernel<<<grid, block>>>(d_sino, d_vol, g);
}

#ifndef MI_NO_MAIN
int main() {
    const MiGeom g = mi_default_geom();
    mi_upload_angles_v2(g);   // 一次性把角度表推进常量内存

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
        mi_launch_v2(d_sino, d_vol, g);
    }
    MI_CUDA_CHECK(cudaDeviceSynchronize());

    const int   kIters = 50;
    cudaEvent_t start, stop;
    MI_CUDA_CHECK(cudaEventCreate(&start));
    MI_CUDA_CHECK(cudaEventCreate(&stop));
    MI_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIters; ++i) {
        mi_launch_v2(d_sino, d_vol, g);
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
    std::printf("[MI] v2 时间 %.3f ms（%d 次平均，5 次预热）\n", ms, kIters);
    std::printf("[MI] v2 有效带宽 %.1f GB/s（按 nx*ny*nAngles*8 字节计）\n",
                bytes / (ms * 1e6));
    std::printf("[MI] v2 估算算力 %.1f GFLOP/s（按每样本 %d FLOP 计）\n",
                flops / (ms * 1e6), MI_FLOP_PER_SAMPLE);
    std::printf("[MI] v2 中心体素 %.6f，全图均值 %.6f\n",
                h_vol[(g.ny / 2) * g.nx + g.nx / 2], mean);

    MI_CUDA_CHECK(cudaEventDestroy(start));
    MI_CUDA_CHECK(cudaEventDestroy(stop));
    MI_CUDA_CHECK(cudaFree(d_sino));
    MI_CUDA_CHECK(cudaFree(d_vol));
    return 0;
}
#endif
