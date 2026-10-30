// kernels.cuh —— 算子声明（模块边界）
//
// 这是整个工程的「接口清单」：pipeline 只认这三行，不关心谁实现、
// 怎么划分 block、用不用共享内存。想换实现（比如换成纹理取样、换成
// 投影驱动），改 .cu，不动 pipeline。
#pragma once

#include "mi/geometry.h"

// 一维斜坡滤波：每条投影单独做一次高通卷积
//   d_in / d_out : 设备指针，形状 nAngles × nDet
void mi_ramp_filter1d(const float* d_in, float* d_out, int nAngles, int nDet);

// 体素驱动反投影：把滤波后的投影抹回二维体素网格
//   d_sino   : 设备指针，形状 nAngles × nDet
//   d_volume : 设备指针，形状 nx × ny
void mi_backproject_voxel(const float* d_sino, float* d_volume, const MiGeometry& g);

// 设备端双线性取样：定义在 src/kernels/interp.cu。
// 之所以单独一个编译单元，是为了示范「设备函数也能跨模块复用」；
// 代价是需要 CUDA_SEPARABLE_COMPILATION（可重定位设备代码），
// 且不加 -dlto 时无法内联——生产代码通常把它放进头文件里 inline。
__device__ float mi_bilinear_sample(const float* row, float u, int nDet);
