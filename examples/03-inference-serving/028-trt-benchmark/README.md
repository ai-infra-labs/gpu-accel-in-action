# 028 · PyTorch→ONNX→TensorRT：一条推理链路的验证清单

同一条 MLP，三条路径同场比较：PyTorch GPU FP32、TensorRT FP32、TensorRT FP16。
脚本自己导出 ONNX、自己构建 engine、自己对齐数值，不需要外部权重。

## 前置

027（推理链路的概念：为什么要量化/编译）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `trt_mlp_benchmark.py` | 导出 ONNX → 构建 fp32/fp16 engine → 计时与误差对照 | 博文代码块 #10 |

## 运行

```bash
python -m pip install -r ../../requirements/advanced.txt
python trt_mlp_benchmark.py
```

产物（当前目录）：`mlp.onnx`、`mlp_fp32.plan`、`mlp_fp16.plan`。无命令行参数，
batch 固定 16、输入维度 1024。

## 环境要求

- NVIDIA GPU + CUDA Toolkit + **TensorRT 10.x**（脚本用 `get_tensor_name` 等新 API，
  8.x 会直接 AttributeError——那是 029 的场景）
- `pycuda` 用于设备内存分配与 stream
- 依赖层：`requirements/advanced.txt`

## 观察点

- FP16 误差应比 FP32 大一个量级，但仍应在 1e-3 内；两者一样大说明精度标志没生效。
- batch 固定 16：小 batch 下 TensorRT 的优势往往不明显，别把这张表外推。

## 验证状态

**未实测。** 本机无 TensorRT 与 NVIDIA GPU，仅完成语法检查
（pyflakes 仅提示 `pycuda.autoprimaryctx` 为副作用导入）。

需要记录：三条路径的 ms/次、加速比、最大/平均绝对误差、engine 大小、上下文显存。
见 `expected_output.md`。

## 出处

- 博文：[28-tensorrt-inference-acceleration-blog-v2.md](../../../blog/028-TensorRT推理加速实战（上）.md)
- 来源代码块 #10，246 行，md5 `673074f4f60ae9f8e31c2f69b2df8e50`

代码逐字一致。文件名沿用博文 docstring 里写的 `trt_mlp_benchmark.py`。
