// utils.h —— 合成数据生成（主机端）
//
// 全程解析生成，不读任何文件：没有 DICOM、没有 NIfTI、没有 ITK。
// 好处是端到端能跑通，且每次结果完全一致，便于对比不同实现的输出。
#pragma once

#include <cstddef>

#include "mi/geometry.h"

// 生成二维 Phantom（椭圆叠加的灰度图），写入 host 缓冲，形状 nx × ny
void mi_gen_phantom_host(float* img, const MiGeometry& g);

// 生成 sinogram（上述椭圆的解析线积分），写入 host 缓冲，形状 nAngles × nDet
void mi_gen_sinogram_host(float* sino, const MiGeometry& g);
