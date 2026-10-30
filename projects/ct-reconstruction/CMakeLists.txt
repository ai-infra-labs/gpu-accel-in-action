cmake_minimum_required(VERSION 3.18)
project(mi_fdk_demo LANGUAGES CXX CUDA)

# 极简工程示例：宁可简陋，也不要库化。
# 目标只有一个可执行文件 fdk_demo，把「滤波 → 反投影」串起来。
#
#   mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc) && ./fdk_demo

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设备函数跨编译单元调用（bilinear_sample 定义在 interp.cu）需要可重定位设备代码
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# 架构：CMake ≥ 3.24 用 native 探测本机卡；老版本回退到常见档位。
# 注意 native 需要 3.24+，所以不能用 cmake_minimum_required(3.18) 就无脑写 native。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
    endif()
endif()

add_executable(fdk_demo
    apps/fdk_demo.cu
    src/pipeline.cu
    src/utils.cu
    src/kernels/backproj.cu
    src/kernels/filter.cu
    src/kernels/interp.cu
)

target_include_directories(fdk_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Release 下开 -O3；教学示例不开 --use_fast_math，避免实测数字对不上 Track A
target_compile_options(fdk_demo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
