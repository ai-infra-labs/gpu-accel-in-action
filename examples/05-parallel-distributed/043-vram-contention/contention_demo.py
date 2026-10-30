"""模型都放进显存了，为什么反而更慢？（第5章 · 并行与分布式）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch。
"""
from statistics import median
from time import perf_counter

import torch
from torch import nn


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要支持 CUDA 的 PyTorch 和可用 NVIDIA GPU。")
    torch.manual_seed(7)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)

    def make_model():
        return nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 1024)
        ).to(device).eval()

    model_a, model_b = make_model(), make_model()
    x_a = torch.randn(128, 1024, device=device)
    x_b = torch.randn(128, 1024, device=device)
    torch.cuda.synchronize(device)  # 初始化就绪，输入和权重保持存活。

    def execute(mode):
        if mode == "A":
            return (model_a(x_a),)
        if mode == "B":
            return (model_b(x_b),)
        if mode == "serial":
            return model_a(x_a), model_b(x_b)
        if mode == "parallel":
            with torch.cuda.stream(stream_a):
                y_a = model_a(x_a)
            with torch.cuda.stream(stream_b):
                y_b = model_b(x_b)
            return y_a, y_b
        raise ValueError(mode)

    @torch.inference_mode()
    def measure(mode):
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        outputs = execute(mode)
        # 设备级等待包含两条分支；结果保留到等待结束之后。
        torch.cuda.synchronize(device)
        elapsed_ms = (perf_counter() - start) * 1000
        extra_peak = torch.cuda.max_memory_allocated(device) - baseline
        del outputs
        return elapsed_ms, extra_peak / (1024 ** 2)

    modes = ["A", "B", "serial", "parallel"]
    for _ in range(5):
        for mode in modes:
            measure(mode)

    samples = {mode: [] for mode in modes}
    for index in range(20):
        offset = index % len(modes)
        for mode in modes[offset:] + modes[:offset]:
            samples[mode].append(measure(mode))

    # 同样的模型和输入，两种调度的输出应在容差内一致。
    with torch.inference_mode():
        expected = execute("serial")
        actual = execute("parallel")
        torch.cuda.synchronize(device)
        for reference, result in zip(expected, actual):
            torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-5)
    del expected, actual

    print("GPU:", torch.cuda.get_device_name(device))
    print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
    times = {}
    for mode in modes:
        times[mode] = median(row[0] for row in samples[mode])
        # 内存使用最差样本，与耗时中位数使用不同汇总口径。
        peak_mib = max(row[1] for row in samples[mode])
        print(f"{mode}: 墙钟中位数 {times[mode]:.3f} ms; "
              f"额外 allocated 峰值最大值 {peak_mib:.2f} MiB")
    print(f"顺序/并发完成时间比: {times['serial'] / times['parallel']:.3f}")
    print("两种调度的输出校验通过。")


if __name__ == "__main__":
    main()
