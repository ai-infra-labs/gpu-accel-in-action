// ============================================================================
// test.cu —— 三层递进的正确性校验（v0 为 golden）
//
// 判定口径（全仓库统一）：逐元素比较
//     |a - b| / (|b| + 1e-12) < 1e-5
// 其中 b 取 v0 的结果。容差放宽到 1e-5 而不是 0，是因为 v2 把
// 「除以 detSpacing」改成「乘以预存好的 1/detSpacing」，末位必然有差异，
// 那属于浮点结合律的正常误差，不是算法分歧。
//
// 三个版本编进同一个翻译单元（各自用 MI_NO_MAIN 关掉自己的 main），
// 因此校验的是真正要跑的那份 kernel，而不是复制出来的副本。
//
// 编译运行： nvcc -O3 -arch=native test.cu -o test && ./test
// ============================================================================
#define MI_NO_MAIN 1
#include "v0_naive.cu"
#include "v1_optimized.cu"
#include "v2_industrial.cu"

#include <algorithm>

// 逐元素相对误差比较，返回 {不合格个数, 最大相对误差}
static void mi_compare(const std::vector<float>& got,
                       const std::vector<float>& golden,
                       const char* name,
                       int& badCount,
                       double& maxRel) {
    badCount = 0;
    maxRel = 0.0;
    const double kTol = 1e-5;
    for (size_t i = 0; i < golden.size(); ++i) {
        const double b = static_cast<double>(golden[i]);
        const double a = static_cast<double>(got[i]);
        const double rel = std::fabs(a - b) / (std::fabs(b) + 1e-12);
        maxRel = std::max(maxRel, rel);
        if (rel >= kTol) {
            ++badCount;
        }
    }
    std::printf("[MI] %s vs golden: 超差 %d / %zu，最大相对误差 %.3e\n",
                name, badCount, golden.size(), maxRel);
}

int main() {
    const MiGeom g = mi_default_geom();
    mi_upload_angles_v2(g);   // v2 的角度表推进常量内存

    std::vector<float> h_sino;
    mi_build_sinogram(h_sino, g);

    const size_t nVox  = static_cast<size_t>(g.nx) * g.ny;
    const size_t nSino = h_sino.size();

    float* d_sino = nullptr;
    MI_CUDA_CHECK(cudaMalloc(&d_sino, nSino * sizeof(float)));
    MI_CUDA_CHECK(cudaMemcpy(d_sino, h_sino.data(), nSino * sizeof(float),
                             cudaMemcpyHostToDevice));

    std::vector<float> h_v0(nVox), h_v1(nVox), h_v2(nVox);
    float* d_vol = nullptr;
    MI_CUDA_CHECK(cudaMalloc(&d_vol, nVox * sizeof(float)));

    // 依次跑三个版本，共用同一块输出缓冲（每次都整块覆盖，不会串味）
    mi_launch_v0(d_sino, d_vol, g);
    MI_CUDA_CHECK(cudaMemcpy(h_v0.data(), d_vol, nVox * sizeof(float),
                             cudaMemcpyDeviceToHost));

    mi_launch_v1(d_sino, d_vol, g);
    MI_CUDA_CHECK(cudaMemcpy(h_v1.data(), d_vol, nVox * sizeof(float),
                             cudaMemcpyDeviceToHost));

    mi_launch_v2(d_sino, d_vol, g);
    MI_CUDA_CHECK(cudaMemcpy(h_v2.data(), d_vol, nVox * sizeof(float),
                             cudaMemcpyDeviceToHost));

    // 先确认 golden 本身不是一潭死水：全零或全 NaN 时，后面的比较毫无意义
    double sum = 0.0;
    size_t nonFinite = 0;
    for (size_t i = 0; i < nVox; ++i) {
        if (!std::isfinite(h_v0[i])) {
            ++nonFinite;
        }
        sum += h_v0[i];
    }
    const double mean = sum / static_cast<double>(nVox);
    std::printf("[MI] golden(v0) 均值 %.6f，非有限值 %zu 个\n", mean, nonFinite);
    if (nonFinite > 0 || std::fabs(mean) < 1e-9) {
        std::printf("[MI] 判定：FAIL —— golden 输出无效，先查数据生成或 kernel\n");
        return 1;
    }

    int bad1 = 0, bad2 = 0;
    double rel1 = 0.0, rel2 = 0.0;
    mi_compare(h_v1, h_v0, "v1", bad1, rel1);
    mi_compare(h_v2, h_v0, "v2", bad2, rel2);

    const bool ok = (bad1 == 0 && bad2 == 0);
    std::printf("[MI] 判定：%s（容差 1e-5，golden = v0）\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
