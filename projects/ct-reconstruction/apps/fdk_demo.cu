// fdk_demo.cu —— 唯一入口
//
// 极简 main：打印参数、调用流水线、打印结束标记。
// 真正的活都在 mi_run_fdk_demo 里，入口不承载任何逻辑。
#include "mi/geometry.h"
#include "mi/pipeline.h"

#include <cstdio>

int main() {
    const MiGeometry g = mi_default_geo2d();

    std::printf("[MI] fdk_demo (synth) nx=%d ny=%d nz=%d nDet=%d nAngles=%d\n",
                g.nx, g.ny, g.nz, g.nDet, g.nAngles);
    mi_run_fdk_demo(g);
    std::printf("[MI] done\n");
    return 0;
}
