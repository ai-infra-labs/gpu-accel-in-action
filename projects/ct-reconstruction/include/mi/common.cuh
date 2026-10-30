// common.cuh —— 错误检查与计时
//
// 这两件事每个 CUDA 项目都要写一遍。工程示例里把它们收成头文件里的两个
// 内联工具，其余模块只调用、不重写。
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 统一的错误检查：失败就打印「文件:行 -> 错误码含义」并退出。
// 不吞异常、不带上下文缺失——出问题时第一眼就能定位。
#define MI_CUDA_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "[MI] CUDA 错误 %s:%d -> %s\n", __FILE__,     \
                         __LINE__, cudaGetErrorString(err__));                 \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// 成对使用：先 record(start)，做活，再 record(stop) + synchronize，
// 最后 mi_time_ms 换算成毫秒。不要把 cudaMemcpy 混进计时窗口，
// 除非你想测的本来就是「含传输的端到端」。
inline void mi_timer_create(cudaEvent_t* start, cudaEvent_t* stop) {
    MI_CUDA_CHECK(cudaEventCreate(start));
    MI_CUDA_CHECK(cudaEventCreate(stop));
}

inline void mi_timer_destroy(cudaEvent_t start, cudaEvent_t stop) {
    MI_CUDA_CHECK(cudaEventDestroy(start));
    MI_CUDA_CHECK(cudaEventDestroy(stop));
}

inline double mi_time_ms(cudaEvent_t start, cudaEvent_t stop) {
    MI_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    MI_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return static_cast<double>(ms);
}
