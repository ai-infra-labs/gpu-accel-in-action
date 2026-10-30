// interp.cu —— 双线性取样（设备端，单独一个编译单元）
//
// 抽出来的理由：反投影、以及将来任何「按浮点坐标取值」的算子都要用它。
// 抽出之后，替换取样策略（最近邻 / 双三次 / 纹理）只改这一处。
//
// 注意：定义在 .cu 而不是头文件里，意味着 depend on 可重定位设备代码
// （CMakeLists.txt 里开了 CUDA_SEPARABLE_COMPILATION）。不加 -dlto 时
// 跨编译单元的设备函数调用不会被内联，比头文件 inline 版本略慢——
// 这是「模块边界」换来的代价，示例里明说，而不是假装没有。
#include "mi/kernels.cuh"

#include <cuda_runtime.h>

__device__ float mi_bilinear_sample(const float* row, float u, int nDet) {
    // 越界夹取：反投影里丢弃等于在图像边缘留黑边
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
