// utils.cu —— 合成数据生成
//
// 全部在主机端解析生成，不读任何文件格式。这样做有两个直接后果：
//   1) 端到端一定跑得通，不依赖数据集与权重；
//   2) 每次运行结果完全一致，不同实现之间可以逐值对比。
#include "mi/utils.h"

#include <cmath>

namespace {

// 若干个椭圆组成 Phantom。参数含义：中心 (cx,cy)、半轴 (a,b)、密度 rho。
// 数值是「看起来像一个横断面」的选择，不对应任何真实病例。
struct Ellipse {
    float cx, cy, a, b, rho;
};

const Ellipse kPhantom[] = {
    { 0.00f,  0.00f, 0.72f, 0.90f,  1.00f},  // 躯干外轮廓
    { 0.00f,  0.00f, 0.60f, 0.78f, -0.35f},  // 内部低密度区
    { 0.30f,  0.10f, 0.22f, 0.30f,  0.45f},  // 病灶 1
    {-0.28f, -0.12f, 0.18f, 0.24f,  0.35f},  // 病灶 2
    { 0.02f, -0.45f, 0.12f, 0.10f,  0.60f},  // 钙化点
};

constexpr int kNumEllipses = static_cast<int>(sizeof(kPhantom) / sizeof(kPhantom[0]));

}  // namespace

void mi_gen_phantom_host(float* img, const MiGeometry& g) {
    const float stepX = 2.0f * g.halfExtent / static_cast<float>(g.nx);
    const float stepY = 2.0f * g.halfExtent / static_cast<float>(g.ny);

    for (int iy = 0; iy < g.ny; ++iy) {
        const float y = (static_cast<float>(iy) + 0.5f) * stepY - g.halfExtent;
        for (int ix = 0; ix < g.nx; ++ix) {
            const float x = (static_cast<float>(ix) + 0.5f) * stepX - g.halfExtent;
            float acc = 0.0f;
            for (int e = 0; e < kNumEllipses; ++e) {
                const Ellipse& el = kPhantom[e];
                const float dx = (x - el.cx) / el.a;
                const float dy = (y - el.cy) / el.b;
                if (dx * dx + dy * dy <= 1.0f) {
                    acc += el.rho;      // 落在椭圆内就叠一层密度
                }
            }
            img[iy * g.nx + ix] = acc;
        }
    }
}

void mi_gen_sinogram_host(float* sino, const MiGeometry& g) {
    // 椭圆在平行束下的线积分有闭式解，因此 sinogram 可以直接算，
    // 不需要前向投影的离散近似，也就没有离散化噪声。
    const float uCenter = 0.5f * static_cast<float>(g.nDet - 1);

    for (int a = 0; a < g.nAngles; ++a) {
        const float theta = static_cast<float>(a) * g.dTheta;
        const float c = cosf(theta);
        const float s = sinf(theta);
        for (int d = 0; d < g.nDet; ++d) {
            const float t = (static_cast<float>(d) - uCenter) * g.detSpacing;
            float acc = 0.0f;
            for (int e = 0; e < kNumEllipses; ++e) {
                const Ellipse& el = kPhantom[e];
                const float tc   = t - (el.cx * c + el.cy * s);
                const float rad2 = el.a * el.a * c * c + el.b * el.b * s * s;
                const float rad  = sqrtf(rad2);
                if (fabsf(tc) < rad) {
                    acc += el.rho * (2.0f * el.a * el.b / rad2) * sqrtf(rad2 - tc * tc);
                }
            }
            sino[static_cast<size_t>(a) * g.nDet + d] = acc;
        }
    }
}
