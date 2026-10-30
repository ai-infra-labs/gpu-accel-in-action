// ============================================================================
// v0_naive.cu —— CT 反投影（Backprojection）的朴素实现
//
// 教学定位：把「反投影」这件事翻译成最直白的 CUDA 写法——
//   一个线程负责一个体素，该体素把所有投影角度扫一遍，逐角度取值累加。
//
// 几何是二维平行束的教学近似（不是生产级 FDK）：
//   体素 (ix,iy) -> 物理坐标 (x,y)
//   角度 θ 下，该点落在探测器上的位置 p = x·cosθ + y·sinθ
//   把 p 换算成探测器索引 u，在 sinogram[a][·] 上做一次双线性取样
//   全部角度累加后乘 dθ 归一化
//
// 编译运行： nvcc -O3 -arch=native v0_naive.cu -o v0_naive && ./v0_naive
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

// ---------------------------------------------------------------------------
// 几何参数：三个版本共用。test.cu 会把三份实现编进同一个翻译单元，
// 所以用宏守卫保证只定义一次（单独编译任一版本时行为不变）。
// ---------------------------------------------------------------------------
#ifndef MI_GEOM_DEFINED
#define MI_GEOM_DEFINED
struct MiGeom {
    int   nx;          // 体素列数（x 方向）
    int   ny;          // 体素行数（y 方向）
    int   nDet;        // 探测器像元数
    int   nAngles;     // 投影角度数
    float halfExtent;  // 空间半宽：x, y ∈ [-halfExtent, +halfExtent]
    float detSpacing;  // 探测器像元间距
    float dTheta;      // 相邻角度间隔（弧度）
};

inline MiGeom mi_default_geom() {
    MiGeom g;
    g.nx = 256;
    g.ny = 256;
    g.nDet = 384;
    g.nAngles = 180;
    g.halfExtent = 1.0f;
    // 探测器要比物体宽一些，否则物体边缘会被切掉
    g.detSpacing = 2.6f / static_cast<float>(g.nDet);
    g.dTheta = 3.14159265358979323846f / static_cast<float>(g.nAngles);
    return g;
}
#endif

// ---------------------------------------------------------------------------
// 合成数据：解析法生成 sinogram（不引入 DICOM / 任何文件格式）
//
// 物体由若干椭圆叠加而成，椭圆 (cx,cy,a,b,rho) 在角度 θ、探测器偏移 t 处的
// 线积分有闭式解，因此 sinogram 可以直接算出来，不需要前向投影的离散近似，
// 也就没有离散化噪声。这一步只在主机端做一次，不计入 kernel 计时。
// ---------------------------------------------------------------------------
#ifndef MI_SHARED_HOST
#define MI_SHARED_HOST
struct MiEllipse {
    float cx, cy, a, b, rho;  // 中心、长短半轴、密度
};

inline const MiEllipse* mi_phantom(int& count) {
    static const MiEllipse kEllipses[] = {
        { 0.00f,  0.00f, 0.72f, 0.90f,  1.00f},  // 躯干外轮廓
        { 0.00f,  0.00f, 0.60f, 0.78f, -0.35f},  // 内部低密度区
        { 0.30f,  0.10f, 0.22f, 0.30f,  0.45f},  // 病灶 1
        {-0.28f, -0.12f, 0.18f, 0.24f,  0.35f},  // 病灶 2
        { 0.02f, -0.45f, 0.12f, 0.10f,  0.60f},  // 钙化点
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
                // 椭圆中心在探测器方向上的偏移
                const float tc = t - (el.cx * c + el.cy * s);
                // 该角度下椭圆的等效半宽
                const float rad2 = el.a * el.a * c * c + el.b * el.b * s * s;
                const float rad = sqrtf(rad2);
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

// ---------------------------------------------------------------------------
// 每个采样点按 14 次浮点运算估算：投影 3 + 索引换算 2 + 双线性 9
// （4 乘 3 加 1 次累加，再摊平取整与比较）。用于把耗时换算成 GFLOP/s。
// ---------------------------------------------------------------------------
#define MI_FLOP_PER_SAMPLE 14

// 朴素双线性取样：不做只读缓存提示，不合并任何访存，完全按直觉写。
__device__ __forceinline__ float bp_sample_v0(const float* row, float u, int nDet) {
    if (u <= 0.0f) {
        return row[0];                       // 落在探测器左边界外，取边缘值
    }
    const float last = static_cast<float>(nDet - 1);
    if (u >= last) {
        return row[nDet - 1];                // 落在右边界外，取边缘值
    }
    const int   i0 = static_cast<int>(u);
    const float f  = u - static_cast<float>(i0);
    // 两个相邻探测器像元按距离加权
    return row[i0] * (1.0f - f) + row[i0 + 1] * f;
}

__global__ void bp_v0_kernel(const float* __restrict__ sino,
                             float* __restrict__ vol,
                             MiGeom g) {
    // 一维线程号摊平到二维体素网格上
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= g.nx * g.ny) {
        return;
    }

    // 摊平索引还原成二维坐标：一次整数除、一次取模
    const int   ix = idx % g.nx;
    const int   iy = idx / g.nx;
    const float stepX = 2.0f * g.halfExtent / static_cast<float>(g.nx);
    const float stepY = 2.0f * g.halfExtent / static_cast<float>(g.ny);
    const float x = (static_cast<float>(ix) + 0.5f) * stepX - g.halfExtent;
    const float y = (static_cast<float>(iy) + 0.5f) * stepY - g.halfExtent;

    const float uCenter = 0.5f * static_cast<float>(g.nDet - 1);
    float sum = 0.0f;
    for (int a = 0; a < g.nAngles; ++a) {
        const float theta = static_cast<float>(a) * g.dTheta;
        // 三角函数每个体素每个角度都重算一次：朴素写法最贵的一笔
        const float p = x * cosf(theta) + y * sinf(theta);
        const float u = p / g.detSpacing + uCenter;
        // 每轮都重新算一次本角度 sinogram 行的首地址
        sum += bp_sample_v0(sino + static_cast<size_t>(a) * g.nDet, u, g.nDet);
    }
    vol[idx] = sum * g.dTheta;
}

void mi_launch_v0(const float* d_sino, float* d_vol, MiGeom g) {
    const int total = g.nx * g.ny;
    const int block = 256;
    const int grid  = (total + block - 1) / block;
    bp_v0_kernel<<<grid, block>>>(d_sino, d_vol, g);
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

    // 预热：首次启动包含上下文与指令缓存的冷启动成本，不计入统计
    for (int i = 0; i < 5; ++i) {
        mi_launch_v0(d_sino, d_vol, g);
    }
    MI_CUDA_CHECK(cudaDeviceSynchronize());

    const int    kIters = 50;
    cudaEvent_t  start, stop;
    MI_CUDA_CHECK(cudaEventCreate(&start));
    MI_CUDA_CHECK(cudaEventCreate(&stop));
    MI_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIters; ++i) {
        mi_launch_v0(d_sino, d_vol, g);
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

    // 有效带宽：每个采样点只读 2 个相邻 float（双线性需要两个抽头）
    const double bytes = static_cast<double>(nVox) * g.nAngles * 2.0 * sizeof(float);
    const double flops = static_cast<double>(nVox) * g.nAngles * MI_FLOP_PER_SAMPLE;

    std::printf("[MI] 规模 nx=%d ny=%d nDet=%d nAngles=%d\n",
                g.nx, g.ny, g.nDet, g.nAngles);
    std::printf("[MI] v0 时间 %.3f ms（%d 次平均，5 次预热）\n", ms, kIters);
    std::printf("[MI] v0 有效带宽 %.1f GB/s（按 nx*ny*nAngles*8 字节计）\n",
                bytes / (ms * 1e6));
    std::printf("[MI] v0 估算算力 %.1f GFLOP/s（按每样本 %d FLOP 计）\n",
                flops / (ms * 1e6), MI_FLOP_PER_SAMPLE);
    std::printf("[MI] v0 中心体素 %.6f，全图均值 %.6f\n",
                h_vol[(g.ny / 2) * g.nx + g.nx / 2], mean);

    MI_CUDA_CHECK(cudaEventDestroy(start));
    MI_CUDA_CHECK(cudaEventDestroy(stop));
    MI_CUDA_CHECK(cudaFree(d_sino));
    MI_CUDA_CHECK(cudaFree(d_vol));
    return 0;
}
#endif
