// pipeline.h —— 流水线入口
//
// 一个函数就是一条流水线：生成数据 → 滤波 → 反投影 → 取一行做体检。
// 入口（apps/fdk_demo.cu）只需要这一行声明，不需要知道里面有几个 kernel。
#pragma once

#include "mi/geometry.h"

void mi_run_fdk_demo(const MiGeometry& g);
