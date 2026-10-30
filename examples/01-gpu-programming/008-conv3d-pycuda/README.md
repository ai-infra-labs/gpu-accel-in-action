# 008 · PyCUDA 手写 3D 卷积：26 万个输出线程，为什么同一批输入被反复访问

`64×64×64` 的体数据，每个输出体素都要看一遍 `3×3×3` 邻域——26 万个线程，
每个做 27 次乘加，总读取量接近 700 万次。这份朴素实现把输入直接放在全局内存里，
相邻输出体素反复读同一批输入，带宽很快成为瓶颈。

本篇先把"能算对"这件事钉死：一个线程一个输出体素，边界按 same padding 处理，
结果与 NumPy 参考实现对照。

## 前置

007（先建立二维卷积的直觉）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `conv3d_pycuda.py` | 3D 卷积 Kernel + same padding + 100 次重复计时 + NumPy 对照 | 博文代码块 #21 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python conv3d_pycuda.py
```

默认参数：`64×64×64` 输入、`kernel_size=3`、`block=(8,8,4)`（256 线程）、
`grid=(8,8,16)`、重复 100 次，种子 42。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- 26 万个输出体素、每个 27 次乘加：用访存量除以显存带宽，看时间是否对得上。
- 明显超标时，根因是同一批输入被反复读取——这正是共享内存复用的动机。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查。

需要记录：平均 Kernel 时间、NumPy 参考时间、最大绝对误差、输出子区域。
见 `expected_output.md`。

## 出处

- 博文：[08-pycuda-3d-convolution-viral.md](../../../blog/008-PyCUDA三维卷积实战.md)
- 来源代码块 #21，201 行，md5 `a7da61977325b7aa1778771f269c3c21`

代码逐字一致。**博文未指定文件名**，`conv3d_pycuda.py` 为仓库新增命名。
