# 029 · 模型上了 GPU，一次请求的时间都花在哪？

一次 TensorRT 推理的时间不是一个数：Host→Device 拷贝、引擎执行、Device→Host
拷贝、Python 侧开销，各占一段。本篇把同一份分类头在 PyTorch / TensorRT FP32 /
TensorRT FP16 上各跑一遍，只统计引擎执行那一段，并核对三条路径的输出。

## 前置

028（同一类实验，但 API 版本不同）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `trt_build_and_run.py` | 导出 ONNX → 构建 fp32/fp16 engine → 计时 → 误差核对 | 博文代码块 #12 |

## 运行

```bash
python -m pip install -r ../../requirements/advanced.txt
python trt_build_and_run.py
```

产物：`classifier.onnx`、`classifier_{fp32,fp16}.plan`。batch 固定 16、输入 1024。

## 环境要求

- NVIDIA GPU + **TensorRT 8.x**：脚本用的是 `get_binding_index` / `num_bindings`
  这套绑定式 API。装的是 10.x 会在这里报错，那是 028 的场景。
- `pycuda.autoprimaryctx`：与 PyTorch 共用 primary context，避免两套上下文互相不可见
- 依赖层：`requirements/advanced.txt`

## 观察点

- FP32 与 FP16 用的容差不同（1e-4 vs 1e-2）——用 1e-4 判 FP16 必然是「未通过」。
- GPU 计算时间不含 H2D/D2H；端到端要另算。

## 验证状态

**未实测。** 本机无 TensorRT，仅完成语法检查。

需要记录：PyTorch ms/次、TensorRT FP32/FP16 ms/次、相对加速比、最大/平均误差、
上下文设备内存、IO 缓冲区字节数。见 `expected_output.md`。

## 出处

- 博文：[29-tensorrt-inference-acceleration-blog.md](../../../blog/029-TensorRT推理加速实战（下）.md)
- 来源代码块 #12，263 行，md5 `ee7333a581d545a804e834b68d2727be`

代码逐字一致。**博文未指定文件名**，`trt_build_and_run.py` 为仓库新增命名。
