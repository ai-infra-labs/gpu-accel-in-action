# 006 · 裁剪不是"切一刀"：用 PyCUDA 讲透 ROI 坐标映射

`image[:, y:y+h, x:x+w]` 在 NumPy 里一行就写完了，GPU 上却要先回答三个问题：
输出线程的二维编号怎么来？它对应输入图的哪个像素？越界线程怎么处理？

本篇用一个 512×512 输入、100×100 的 ROI，把"输出坐标 → 输入坐标"的偏移量
算清楚，并用 `np.array_equal` 逐字节核对（纯复制没有浮点误差，用 `allclose`
反而放过了错误）。

## 前置

005（同样是二维线程映射）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `image_crop_roi.py` | ROI 参数校验 + 裁剪 Kernel + Kernel/端到端双口径计时 | 博文代码块 #18 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python image_crop_roi.py
```

默认参数：输入 `512×512`、ROI `100×100 @ (100,100)`、`block=(16,16,1)`、
`grid=(7,7)`。脚本在模块级执行（无 `main()`）。
计时口径：warmup×5 + repeat×20，取 median 与 P95。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- 端到端时间会比 Kernel 时间大一个量级：2 MiB 输入的 H2D 一人就占几十微秒。
- grid=(7,7) 会多启动 12×12 个线程，靠边界判断挡掉；去掉判断就会踩到相邻像素。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查。

需要记录：Kernel 的 median/P95、端到端的 median/P95、逐字节一致性。
见 `expected_output.md`。

## 出处

- 博文：[06-cuda-image-cropping-pycuda-light-tech.md](../../../blog/006-CUDA图像裁剪（PyCUDA实战）.md)
- 来源代码块 #18，205 行，md5 `0626e49110352e06225653bf7d849c69`

代码逐字一致。**博文未指定文件名**，`image_crop_roi.py` 为仓库新增命名。
