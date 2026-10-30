// backproj.cu —— 体素驱动反投影（从教学版抽取的主干）
//
// 与 Track A 的 v1 是同一个思路：一个线程一个体素，扫全部角度累加；
// 这里去掉了大段教学注释，只留索引计算的主干，方便对照。
//
// 取样走 mi_bilinear_sample（另一个编译单元），体现「算子拼接」而非单文件演示。
#include "mi/kernels.cuh"

#include <cuda_runtime.h>

namespace {

constexpr int kBlockX = 32;   // 一个 warp 覆盖同一行相邻 32 列
constexpr int kBlockY = 8;

__global__ void backproj_kernel(const float* __restrict__ sino,
                                float* __restrict__ volume,
                                MiGeometry g) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= g.nx || iy >= g.ny) {
        return;
    }

    // 体素索引 → 物理坐标
    const float stepX = 2.0f * g.halfExtent / static_cast<float>(g.nx);
    const float stepY = 2.0f * g.halfExtent / static_cast<float>(g.ny);
    const float x = (static_cast<float>(ix) + 0.5f) * stepX - g.halfExtent;
    const float y = (static_cast<float>(iy) + 0.5f) * stepY - g.halfExtent;

    const float uCenter = 0.5f * static_cast<float>(g.nDet - 1);
    const float* row = sino;
    float sum = 0.0f;
    for (int a = 0; a < g.nAngles; ++a, row += g.nDet) {
        const float theta = static_cast<float>(a) * g.dTheta;
        const float p = x * cosf(theta) + y * sinf(theta);
        const float u = p / g.detSpacing + uCenter;
        sum += mi_bilinear_sample(row, u, g.nDet);
    }
    volume[iy * g.nx + ix] = sum * g.dTheta;
}

}  // namespace

void mi_backproject_voxel(const float* d_sino, float* d_volume, const MiGeometry& g) {
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((g.nx + block.x - 1) / block.x,
                    (g.ny + block.y - 1) / block.y);
    backproj_kernel<<<grid, block>>>(d_sino, d_volume, g);
}
