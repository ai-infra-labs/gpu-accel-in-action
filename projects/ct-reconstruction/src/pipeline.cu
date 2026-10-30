// pipeline.cu —— 把「滤波 → 反投影」串起来
//
// 这一层的职责只有三个：
//   1) 准备数据（合成，全程内存里生成）；
//   2) 按顺序调用两个算子，边界上只传设备指针；
//   3) 计时并打印一个最小校验量（中心行均值）。
//
// 它不碰 block 划分、不碰 shared memory、不碰索引计算——那些属于 kernels/。
// 这就是「职责边界」在代码里的样子：pipeline 改了不会影响 kernel，
// kernel 换了实现也不会影响 pipeline。
#include "mi/pipeline.h"

#include "mi/common.cuh"
#include "mi/kernels.cuh"
#include "mi/utils.h"

#include <cmath>
#include <cstdio>
#include <vector>

void mi_run_fdk_demo(const MiGeometry& g) {
    const size_t nSino = static_cast<size_t>(g.nAngles) * g.nDet;
    const size_t nVox  = static_cast<size_t>(g.nx) * g.ny;

    // ---- 1. 主机端合成数据 ----
    // phantom 是「物体本来长什么样」的参考图，本流水线不用它做数值校验
    // （未滤波的反投影与 phantom 不存在逐值关系），只打印它的中心行均值，
    // 让人一眼确认数据生成本身不是一潭死水。
    std::vector<float> h_phantom(nVox);
    std::vector<float> h_sino(nSino);
    mi_gen_phantom_host(h_phantom.data(), g);
    mi_gen_sinogram_host(h_sino.data(), g);

    float* d_sino = nullptr;   // 原始投影
    float* d_filt = nullptr;   // 滤波后投影
    float* d_vol  = nullptr;   // 重建结果
    MI_CUDA_CHECK(cudaMalloc(&d_sino, nSino * sizeof(float)));
    MI_CUDA_CHECK(cudaMalloc(&d_filt, nSino * sizeof(float)));
    MI_CUDA_CHECK(cudaMalloc(&d_vol, nVox * sizeof(float)));
    MI_CUDA_CHECK(cudaMemcpy(d_sino, h_sino.data(), nSino * sizeof(float),
                             cudaMemcpyHostToDevice));

    // ---- 2. 两个算子串联，计时只包住这一段 ----
    // 口径：从「数据已在设备上」到「重建结果在设备上」，
    // 不含 H2D/D2H。写文章时别把这段和单算子 kernel 时间混为一谈。
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    mi_timer_create(&start, &stop);
    MI_CUDA_CHECK(cudaEventRecord(start));
    mi_ramp_filter1d(d_sino, d_filt, g.nAngles, g.nDet);
    mi_backproject_voxel(d_filt, d_vol, g);
    MI_CUDA_CHECK(cudaEventRecord(stop));
    const double ms = mi_time_ms(start, stop);

    // ---- 3. 取回结果，做最小体检 ----
    std::vector<float> h_vol(nVox);
    MI_CUDA_CHECK(cudaMemcpy(h_vol.data(), d_vol, nVox * sizeof(float),
                             cudaMemcpyDeviceToHost));

    const int    row    = g.ny / 2;
    const size_t rowOff = static_cast<size_t>(row) * g.nx;
    double sum = 0.0;
    size_t nonFinite = 0;
    for (int ix = 0; ix < g.nx; ++ix) {
        const float v = h_vol[rowOff + ix];
        if (!std::isfinite(v)) {
            ++nonFinite;
        }
        sum += v;
    }
    const double rowMean = sum / static_cast<double>(g.nx);

    // phantom 的中心行均值：只用于说明「数据非空」，不是重建质量指标
    double phantomSum = 0.0;
    for (int ix = 0; ix < g.nx; ++ix) {
        phantomSum += h_phantom[rowOff + ix];
    }
    const double phantomMean = phantomSum / static_cast<double>(g.nx);

    std::printf("[MI] FDK pipeline (synth) time: %.3f ms\n", ms);
    std::printf("[MI] center row mean: %.6f (count=%d)\n", rowMean, g.nx);
    std::printf("[MI] phantom center row mean: %.6f（参考值，非质量指标）\n",
                phantomMean);
    if (nonFinite > 0) {
        std::printf("[MI] 警告：中心行有 %zu 个非有限值，重建结果不可信\n", nonFinite);
    }

    MI_CUDA_CHECK(cudaFree(d_sino));
    MI_CUDA_CHECK(cudaFree(d_filt));
    MI_CUDA_CHECK(cudaFree(d_vol));
    mi_timer_destroy(start, stop);
}
