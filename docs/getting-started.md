# 构建：三条路径，别混着用

仓库里的 CUDA 代码不是一种东西，硬塞进一个构建系统只会互相打架。按代码的形态分三条路径：

| 路径 | 覆盖范围 | 构建方式 | 为什么不能合并 |
|---|---|---|---|
| **单文件教学单元** | `projects/ct-reconstruction/ops/*/*.cu`（6 个） | 顶层 `CMakeLists.txt` | 自带 `main()`、不依赖外部头文件，一把 `nvcc` 就够 |
| **CMake 工程示例** | `projects/ct-reconstruction/{src,include,apps}/` | 该目录自己的 `CMakeLists.txt` | 跨编译单元调用设备函数，需要 `CUDA_SEPARABLE_COMPILATION` 与 include 路径 |
| **torch 扩展** | `06-custom-operators/056-*/affine_kernel.cu` | 由同目录 `bench_three_ways.py` 现场 JIT | 依赖 `ATen/ATen.h`、`c10/`、pybind11，裸 `nvcc` 找不到这些头文件 |

## 一条命令编全部单文件单元

```bash
cmake -B build && cmake --build build -j        # 产物在 build/bin/
ctest --test-dir build                          # 只有 backprojection_test 会跑（它用退出码表达 PASS/FAIL）
```

指定架构（默认 `native`，即探测本机）：

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90    # H100
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89    # RTX 40 系
```

工程示例单独编：

```bash
cmake -B build-fdk -S projects/ct-reconstruction && cmake --build build-fdk -j
./build-fdk/fdk_demo
```

torch 扩展不需要手工编译，跑它的基准脚本即可：

```bash
cd 06-custom-operators/056-custom-op-vs-eager && python bench_three_ways.py
```

## 架构：为什么不用写死的 `sm_86`

`-arch=sm_86` 只生成 8.6 的 SASS，**不嵌 PTX**。换到 H100（9.0）或 B200 上会直接报：

```
no kernel image is available for execution on the device
```

顶层 `CMakeLists.txt` 因此默认 `CMAKE_CUDA_ARCHITECTURES=native`（需要 CMake ≥ 3.24、CUDA ≥ 11.5）。
如果目标机器没有卡又要预编译（比如打 CI 镜像），用显式双 code 保留 PTX：

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="80-real;90-real"   # 分别生成 8.0/9.0 SASS
```

真要跨代通用，加一个 `-gencode arch=compute_90,code=compute_90` 让驱动 JIT 兜底，
代价是首次启动慢几百毫秒。

## 验证状态

**这套构建入口尚未在 N 卡机器上实跑过。** 编写环境无 `nvcc`，只做了静态检查：

- 目标清单来自对 6 个 `ops/*/*.cu` 的 `main()` 与本地 `#include` 扫描；
- 顶层 CMake 未参与 `make check`（CI 无 CUDA Toolkit），所以它坏了不会让 CI 变红；
- `docs/VERIFY.md` 里这些篇目的状态仍是「未实测」，本次改动没有改变任何实测结论。

第一次在有卡的机器上跑通后，请把 `cmake` 版本、GPU 型号、`ctest` 输出补到
`projects/ct-reconstruction/docs/expected_output.md`，并按
[验证状态](reference/verification-status.md) 的流程升级状态。

## 常见失败

| 现象 | 原因 | 处理 |
|---|---|---|
| `no kernel image is available` | 架构写死且目标卡不同代 | 用 `native` 或补 PTX |
| `fatal error: ATen/ATen.h: No such file` | 拿 torch 扩展当普通 `.cu` 编 | 走 `bench_three_ways.py` |
| `undefined reference to __cudaRegisterLinkedBinary` | 设备函数跨文件却没开 separable compilation | 该工程加 `CMAKE_CUDA_SEPARABLE_COMPILATION ON` |
| 链接期 `multiple definition` | 两个 `.cu` 各自定义了同名 host 辅助函数 | 用 `static`/匿名命名空间收敛到单文件 |
