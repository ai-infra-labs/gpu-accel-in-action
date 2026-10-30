# 015 · 把声音拆成频率，究竟发生了什么

离散傅里叶变换（DFT）的定义只有一行：每个频率桶 $k$ 都要跟全部 $N$ 个采样
做一次加权求和，权重是旋转因子 $e^{-2\pi i kn/N}$。朴素实现因此要做 $N^2$ 次
复数乘加——`N=1024` 时是 104 万次。

本篇用"一个线程一个频率桶"的朴素 DFT 把这件事跑通，并用 FFT 做对照：
同一份信号，两条路径给出的频谱是否一致。

## 前置

010（共享内存搬运与同步）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `naive_dft.py` | 朴素 DFT Kernel + 自检（4 种规模 × 3 类信号）+ 双音频谱 + 基准 | 博文代码块 #28 |

## 运行

```bash
# 自检：64/256/1024/4096 四个规模 × 三类信号，全部必须 PASS
python -m pip install -r ../../requirements/python-gpu.txt
python naive_dft.py --self-test

# 演示：1024 点、采样率 8192 Hz、440 Hz + 1000 Hz 双音
python naive_dft.py
```

默认参数：`n=1024`、`sample_rate=8192.0`、`block_size=256`。
频谱幅度按 $2/N$ 标定，所以双音的峰值应当落在 440 Hz 与 1000 Hz 附近
（频率分辨率 $8192/1024 = 8$ Hz）。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit（`SourceModule` 带 `-O3`）
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- 落桶信号（440/1000 Hz 在 fs/N=8 Hz 下都是整数桶）幅度应≈1.0；非落桶会因泄漏低于 1。
- 大 N 时误差变大，先怀疑 sincosf 的角度精度，而不是 FFT 本身。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查。

需要记录：自检是否全 PASS、最大绝对误差、相对 L2 误差、前 5 个峰值频点与幅度、
kernel-only 基准。见 `expected_output.md`。

## 出处

- 博文：[15-gpu-naive-dft-fourier-light-tech.md](../../../blog/015-GPU朴素DFT与傅里叶变换入门.md)
- 来源代码块 #28，300 行，md5 `b71b5c16a2af41e525801f82836a8009`

代码逐字一致。**博文未指定文件名**，`naive_dft.py` 为仓库新增命名。
