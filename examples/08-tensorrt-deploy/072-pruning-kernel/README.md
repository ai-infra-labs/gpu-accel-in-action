# 072 · 给权重做剪枝，CUDA 到底加速了哪一步

"名字里带 `cuda`，计算却回到 CPU"是剪枝代码里最常见的写法：权重在显存，
`param.data.cpu().numpy()` 一调，整张表搬到主机、在 NumPy 里比阈值、再搬回去。
筛选确实做完了，但多付了两次跨总线拷贝。

本篇把三件事分开：

| 问题 | 脚本 |
|---|---|
| 往返写法到底贵不贵？贵在哪一段？ | `torch_prune_roundtrip_demo.py` |
| GPU 上原地筛选要写多少代码？ | `threshold_prune_demo.py`（PyCUDA 自定义 kernel） |
| 剪枝后推理会不会变快？ | **本篇不回答**——那是第 073 篇的账 |

## 前置

071、060（量化与显存记账）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `torch_prune_roundtrip_demo.py` | PyTorch 两条路径（GPU→CPU→GPU vs 权重留 GPU）+ p50/p95 | 博文代码块 #4 |
| `threshold_prune_demo.py` | PyCUDA 阈值 kernel，含 CPU / GPUArray / kernel / 往返四种计时 | 博文代码块 #7 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python torch_prune_roundtrip_demo.py    # 需要 torch + CUDA
python threshold_prune_demo.py          # 需要 PyCUDA，独立建立 context
```

两个脚本各自独立，不要同时跑：`threshold_prune_demo.py` 会自己
`import pycuda.autoinit` 建上下文，混用其他框架的指针会出问题。

## 环境要求

- 必须：NVIDIA GPU
  - `torch_prune_roundtrip_demo.py`：torch 的 CUDA 版本
  - `threshold_prune_demo.py`：PyCUDA + `nvcc`
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- p50 比值（往返 vs 驻留）就是要的结论；两块代码的输出必须逐值一致。
- 剪枝快不等于推理快：本篇只测筛选那一步。

## 验证状态

**未实测。** 本机无 `nvcc`、无 CUDA 版 torch，仅完成语法检查。

需要记录：两条路径 p50/p95 与比值、四种计时的对比、CPU/GPU 结果一致性。
见 `expected_output.md`。

## 出处

- 博文：[72-给权重做剪枝CUDA到底加速了哪一步.md](../../../blog/072-给权重做剪枝CUDA到底加速了哪一步.md)
- 代码块 #4，84 行，md5 `1117f23f38e049a1d0fc3222dbdf8b4e` → `torch_prune_roundtrip_demo.py`
- 代码块 #7，111 行，md5 `2fa23ddbfb60a02f2f748b4f0d583515` → `threshold_prune_demo.py`

代码逐字一致。**博文未指定文件名**，两个文件名为仓库新增。
