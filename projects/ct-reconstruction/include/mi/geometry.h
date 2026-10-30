// geometry.h —— 极简几何参数
//
// 集中放一个结构体，避免 nx / ny / nDet / dTheta 这些标量在函数间到处传。
// 这是本示例唯一「像工程」的地方：参数有单一来源，改规模只改一处。
#pragma once

struct MiGeometry {
    int   nx;          // 体素列数（x）
    int   ny;          // 体素行数（y）
    int   nz;          // 体素层数（z），本示例固定 1
    int   nDet;        // 探测器像元数
    int   nAngles;     // 投影角度数
    float halfExtent;  // 空间半宽：x, y ∈ [-halfExtent, +halfExtent]
    float detSpacing;  // 探测器像元间距
    float dTheta;      // 相邻角度间隔（弧度）
};

// 二维平行束的默认几何：规模小到几百毫秒内跑完，足够看清数据流
inline MiGeometry mi_default_geo2d() {
    MiGeometry g{};
    g.nx = 64;
    g.ny = 64;
    g.nz = 1;
    g.nDet = 96;
    g.nAngles = 64;
    g.halfExtent = 1.0f;
    g.detSpacing = 2.6f / static_cast<float>(g.nDet);
    g.dTheta = 3.14159265358979323846f / static_cast<float>(g.nAngles);
    return g;
}
