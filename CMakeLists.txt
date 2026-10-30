cmake_minimum_required(VERSION 3.24)

# gpu-accel-in-action 顶层构建入口。
#
# 只做一件事：把「单文件、自带 main()、nvcc 一把能编」的 .cu 变成可复制的构建命令。
# 仓库里的 CUDA 代码有三条构建路径，这里只覆盖第一条：
#
#   1. 单文件教学单元（projects/ct-reconstruction/ops/*/*.cu）→ 本文件
#   2. CMake 工程示例（projects/ct-reconstruction/src|include|apps/）→ 该目录自己的 CMakeLists.txt
#   3. torch 扩展（06-custom-operators/056-*/affine_kernel.cu）→ 由同目录的 .py 用
#      torch.utils.cpp_extension.load() 现场编译，裸 nvcc 缺 torch 头文件，编不过
#
# 三条路径的取舍与命令见 docs/BUILD.md。
#
# 用法：
#   cmake -B build && cmake --build build -j
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90    # 指定架构（默认 native）

project(cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  # 性能类示例不带优化等于白跑，默认 Release
  set(CMAKE_BUILD_TYPE Release CACHE STRING "构建类型" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # native = 让 nvcc 探测本机算力（需要 CMake ≥3.24 与 CUDA ≥11.5）。
  # 不要退回写死的 sm_86：那样只生成一套 SASS、不嵌 PTX，
  # 换到 H100 / B200 会报 no kernel image is available for execution。
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "目标 GPU 架构，默认 native" FORCE)
endif()

enable_testing()

# ---------------------------------------------------------------- 1. 教学单元
# 每个 ops/<算子>/<版本>.cu 编成一个可执行文件，命名 <算子>_<版本>，例如
# backprojection_v0_naive。这些文件是自足的：不依赖外部头文件，只 include 标准库。
file(GLOB TEACHING_CU CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/projects/ct-reconstruction/ops/*/*.cu")

foreach(src IN LISTS TEACHING_CU)
  get_filename_component(stem "${src}" NAME_WE)
  get_filename_component(dir  "${src}" DIRECTORY)
  get_filename_component(op   "${dir}"  NAME)
  set(target "${op}_${stem}")

  add_executable(${target} "${src}")
  target_include_directories(${target} PRIVATE "${dir}")
  # 单文件之间互不依赖，关掉 cutlass 之类的外部依赖注入的可能
  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

  # 只有 test.cu 会以退出码表达 PASS/FAIL，其余是打表的 benchmark，
  # 进 ctest 会变成「跑一遍就算过」，没有意义。
  if(stem STREQUAL "test")
    add_test(NAME ${target} COMMAND ${target})
  endif()
endforeach()

list(LENGTH TEACHING_CU _n_teaching)
message(STATUS "教学单元：${_n_teaching} 个可执行目标")

# ---------------------------------------------------------------- 2. 工程示例
option(CUDA_KERNELS_BUILD_PROJECT
       "是否把 projects/ct-reconstruction 的 CMake 工程示例一并构建" ON)
if(CUDA_KERNELS_BUILD_PROJECT AND EXISTS
   "${CMAKE_CURRENT_SOURCE_DIR}/projects/ct-reconstruction/CMakeLists.txt")
  add_subdirectory(projects/ct-reconstruction)
endif()

# ---------------------------------------------------------------- 3. 章节目录
# 章节目录下目前只有 056 的 torch 扩展。这里做个探针而不是直接忽略：
# 将来有人往章节目录里放了单文件 .cu，构建时会被明确提示，而不是悄悄漏编。
file(GLOB CHAPTER_CU CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/examples/0[1-9]-*/[0-9][0-9][0-9]-*/*.cu")
foreach(src IN LISTS CHAPTER_CU)
  get_filename_component(name "${src}" NAME)
  if(name STREQUAL "affine_kernel.cu")
    # 056：torch 扩展，编译入口在 bench_three_ways.py（cpp_extension.load）
    continue()
  endif()
  message(WARNING
    "章节目录下的 ${src} 没有接入构建：单文件示例请放到 projects/<章>/ops/ 下，"
    "或在本文件里为它补一条 add_executable。")
endforeach()

message(STATUS "CUDA 架构：${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "构建产物目录：${CMAKE_BINARY_DIR}/bin")
