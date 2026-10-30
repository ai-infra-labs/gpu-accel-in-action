// ============================================================================
// v0_naive.cu —— 双线性插值的最小教学版
//
// 为什么它出现在 CT 反投影里：反投影要按 u = p / detSpacing + 中心 去取值，
// 而 u 几乎不可能是整数。取最近邻会让重建图出现块状台阶，双线性用
// 相邻两个探测器按距离加权，把台阶抹平。
//
// 这里把「取样」单独抽出来做最小示例：给一堆任意浮点坐标，逐点双线性取值，
// 并与 CPU 参考实现逐点对答案。边界统一用「夹取」而不是「丢弃」——
// 反投影里丢弃等于在图像边缘留黑边，夹取至少不会比最外侧探测器更离谱。
//
// 编译运行： nvcc -O3 -arch=native v0_naive.cu -o bilinear && ./bilinear
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

constexpr int kWidth   = 64;    // 图像宽
constexpr int kHeight  = 64;    // 图像高
constexpr int kSamples = 4096;  // 取样点数
constexpr int kBlock   = 128;   // 每个 block 的线程数

// 双线性取样：坐标 (fx, fy) 落在像素中心网格上，(0,0) 是左上角像素中心
__device__ __forceinline__ float bilinear_at(const float* __restrict__ img,
                                             float fx, float fy,
                                             int width, int height) {
    // 越界直接夹到边界，避免读飞
    if (fx < 0.0f) fx = 0.0f;
    if (fy < 0.0f) fy = 0.0f;
    const float maxX = static_cast<float>(width - 1);
    const float maxY = static_cast<float>(height - 1);
    if (fx > maxX) fx = maxX;
    if (fy > maxY) fy = maxY;

    const int   x0 = static_cast<int>(fx);
    const int   y0 = static_cast<int>(fy);
    const int   x1 = (x0 + 1 < width) ? x0 + 1 : x0;    // 右边界退化成一列
    const int   y1 = (y0 + 1 < height) ? y0 + 1 : y0;   // 下边界退化
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);

    // 四个角
    const float v00 = img[y0 * width + x0];
    const float v10 = img[y0 * width + x1];
    const float v01 = img[y1 * width + x0];
    const float v11 = img[y1 * width + x1];

    // 先横向插值两次，再纵向插值一次
    const float top    = v00 + (v10 - v00) * tx;
    const float bottom = v01 + (v11 - v01) * tx;
    return top + (bottom - top) * ty;
}

__global__ void bilinear_kernel(const float* __restrict__ img,
                                const float* __restrict__ xs,
                                const float* __restrict__ ys,
                                float* __restrict__ out,
                                int n, int width, int height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    out[i] = bilinear_at(img, xs[i], ys[i], width, height);
}

// CPU 参考：与 kernel 逐行对应的朴素实现
static float bilinear_cpu(const std::vector<float>& img, float fx, float fy,
                          int width, int height) {
    if (fx < 0.0f) fx = 0.0f;
    if (fy < 0.0f) fy = 0.0f;
    const float maxX = static_cast<float>(width - 1);
    const float maxY = static_cast<float>(height - 1);
    if (fx > maxX) fx = maxX;
    if (fy > maxY) fy = maxY;

    const int   x0 = static_cast<int>(fx);
    const int   y0 = static_cast<int>(fy);
    const int   x1 = (x0 + 1 < width) ? x0 + 1 : x0;
    const int   y1 = (y0 + 1 < height) ? y0 + 1 : y0;
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);

    const float v00 = img[y0 * width + x0];
    const float v10 = img[y0 * width + x1];
    const float v01 = img[y1 * width + x0];
    const float v11 = img[y1 * width + x1];

    const float top    = v00 + (v10 - v00) * tx;
    const float bottom = v01 + (v11 - v01) * tx;
    return top + (bottom - top) * ty;
}

int main() {
    // 合成图像：两个方向的正弦叠加，保证任意方向都有梯度，插值误差才看得出来
    std::vector<float> h_img(static_cast<size_t>(kWidth) * kHeight);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const float u = static_cast<float>(x) / kWidth;
            const float v = static_cast<float>(y) / kHeight;
            h_img[static_cast<size_t>(y) * kWidth + x] =
                sinf(6.2831853f * u) + 0.5f * cosf(4.0f * v);
        }
    }

    // 取样坐标：故意横跨边界（-3 到 66），同时验证夹取逻辑
    std::vector<float> h_xs(kSamples), h_ys(kSamples);
    for (int i = 0; i < kSamples; ++i) {
        const float t = static_cast<float>(i) / kSamples;
        h_xs[i] = -3.0f + 69.0f * t;
        h_ys[i] = 63.0f * (0.5f + 0.5f * sinf(12.566370f * t)) - 2.0f;
    }

    const size_t nImg = h_img.size();
    float* d_img = nullptr;
    float* d_xs  = nullptr;
    float* d_ys  = nullptr;
    float* d_out = nullptr;
    MI_CUDA_CHECK(cudaMalloc(&d_img, nImg * sizeof(float)));
    MI_CUDA_CHECK(cudaMalloc(&d_xs, kSamples * sizeof(float)));
    MI_CUDA_CHECK(cudaMalloc(&d_ys, kSamples * sizeof(float)));
    MI_CUDA_CHECK(cudaMalloc(&d_out, kSamples * sizeof(float)));
    MI_CUDA_CHECK(cudaMemcpy(d_img, h_img.data(), nImg * sizeof(float),
                             cudaMemcpyHostToDevice));
    MI_CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), kSamples * sizeof(float),
                             cudaMemcpyHostToDevice));
    MI_CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), kSamples * sizeof(float),
                             cudaMemcpyHostToDevice));

    const int grid = (kSamples + kBlock - 1) / kBlock;
    bilinear_kernel<<<grid, kBlock>>>(d_img, d_xs, d_ys, d_out,
                                      kSamples, kWidth, kHeight);
    MI_CUDA_CHECK(cudaGetLastError());
    MI_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(kSamples);
    MI_CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, kSamples * sizeof(float),
                             cudaMemcpyDeviceToHost));

    // 先校验再下结论：与 CPU 逐点比对
    double maxAbs = 0.0;
    for (int i = 0; i < kSamples; ++i) {
        const float ref = bilinear_cpu(h_img, h_xs[i], h_ys[i], kWidth, kHeight);
        maxAbs = std::fmax(maxAbs, std::fabs(static_cast<double>(h_out[i]) - ref));
    }

    // 整数坐标处必须精确等于原像素值——这是插值实现最容易写错的地方
    double maxIntErr = 0.0;
    for (int y = 0; y < kHeight; y += 7) {
        for (int x = 0; x < kWidth; x += 9) {
            const float got = bilinear_cpu(h_img, static_cast<float>(x),
                                           static_cast<float>(y), kWidth, kHeight);
            const float ref = h_img[static_cast<size_t>(y) * kWidth + x];
            maxIntErr = std::fmax(maxIntErr, std::fabs(static_cast<double>(got) - ref));
        }
    }

    std::printf("[MI] 规模 %dx%d，取样点 %d，block=%d\n",
                kWidth, kHeight, kSamples, kBlock);
    std::printf("[MI] GPU 与 CPU 最大绝对误差 %.3e\n", maxAbs);
    std::printf("[MI] 整数坐标处最大偏差 %.3e（应接近 0）\n", maxIntErr);
    std::printf("[MI] 判定：%s（容差 1e-6）\n",
                (maxAbs < 1e-6 && maxIntErr < 1e-6) ? "PASS" : "FAIL");

    MI_CUDA_CHECK(cudaFree(d_img));
    MI_CUDA_CHECK(cudaFree(d_xs));
    MI_CUDA_CHECK(cudaFree(d_ys));
    MI_CUDA_CHECK(cudaFree(d_out));
    return (maxAbs < 1e-6 && maxIntErr < 1e-6) ? 0 : 1;
}
