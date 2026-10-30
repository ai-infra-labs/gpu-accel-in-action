"""给权重做剪枝，CUDA 到底加速了哪一步？（第8章 · TensorRT 部署）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch、NumPy。
"""
import time

import numpy as np
import torch
import torch.nn as nn


class DemoNet(nn.Module):
    """演示网络：256 → 128 → 64 → 10。"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)


def build_model(seed=0):
    """每次都用同一个种子构建，保证各次采样的输入完全一致。"""
    torch.manual_seed(seed)
    return DemoNet().cuda()


def prune_via_cpu(model, threshold):
    """开篇那种写法：权重在 GPU，筛选在 CPU 上完成。"""
    for name, param in model.named_parameters():
        if not name.endswith("weight"):
            continue
        weights = param.data.cpu().numpy()      # 设备 → 主机：同步拷贝
        mask = np.abs(weights) > threshold
        weights[~mask] = 0
        # 开篇写的是 .cuda()，隐含默认设备；这里按权重所在设备回传。
        param.data = torch.from_numpy(weights).to(param.device)


def prune_on_gpu(model, threshold):
    """权重留在 GPU，用张量运算完成筛选。"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not name.endswith("weight"):
                continue
            param.masked_fill_(param.abs() <= threshold, 0)


def measure(prune_fn, threshold, warmup=10, samples=50):
    """每次采样都新建模型，保证两条路径处理的是同一份未修改的权重。"""
    for _ in range(warmup):
        prune_fn(build_model(), threshold)
        torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        model = build_model()
        torch.cuda.synchronize()                 # 停表前先确保上一次工作已完成
        start = time.perf_counter()
        prune_fn(model, threshold)
        torch.cuda.synchronize()                 # 等 GPU 真正做完再停表
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95, method="linear"))


def main():
    threshold = 0.125

    # 先比正确性：两条路径必须给出同样的权重。
    model_cpu = build_model()
    model_gpu = build_model()
    prune_via_cpu(model_cpu, threshold)
    prune_on_gpu(model_gpu, threshold)
    for (name_a, pa), (name_b, pb) in zip(model_cpu.named_parameters(),
                                          model_gpu.named_parameters()):
        assert name_a == name_b
        # 演示权重由 PyTorch 随机初始化，不含 NaN，两条路径判定结果一致。
        torch.testing.assert_close(pa.detach(), pb.detach(), rtol=0, atol=0)
    print("两条路径结果一致。")

    p50_cpu, p95_cpu = measure(prune_via_cpu, threshold)
    p50_gpu, p95_gpu = measure(prune_on_gpu, threshold)
    print("原始往返写法（GPU→CPU→GPU）: p50=%.4f ms, p95=%.4f ms" % (p50_cpu, p95_cpu))
    print("GPU 张量筛选（权重留在 GPU）: p50=%.4f ms, p95=%.4f ms" % (p50_gpu, p95_gpu))
    print("p50 比值: %.2fx" % (p50_cpu / p50_gpu))


if __name__ == "__main__":
    main()
