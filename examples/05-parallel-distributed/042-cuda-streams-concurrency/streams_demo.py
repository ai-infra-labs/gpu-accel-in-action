"""开了四个线程，GPU 就会同时跑吗？（第5章 · 并行与分布式）。

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
    main_stream = torch.cuda.current_stream(device)
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)

    def make_model():
        return nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 512)
        ).to(device).eval()

    model_a, model_b = make_model(), make_model()
    x_a = torch.randn(256, 512, device=device)
    x_b = torch.randn(256, 512, device=device)

    # 排除初始化，并确保两条分支能安全使用权重与输入。
    torch.cuda.synchronize(device)

    # 测量用 Event 在 run() 外创建并复用：底层 CUDA Event 是延迟初始化的，
    # 每轮新建会让预热落空，也把事件创建开销混入小模型的对比。
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    done_a = torch.cuda.Event()
    done_b = torch.cuda.Event()

    @torch.inference_mode()
    def run(use_two_streams):
        wall_start = perf_counter()
        start.record(main_stream)

        if use_two_streams:
            with torch.cuda.stream(stream_a):
                stream_a.wait_event(start)
                y_a = model_a(x_a)
                done_a.record(stream_a)
            with torch.cuda.stream(stream_b):
                stream_b.wait_event(start)
                y_b = model_b(x_b)
                done_b.record(stream_b)

            # 主队列先汇合两条分支，再打终点标记。
            main_stream.wait_event(done_a)
            main_stream.wait_event(done_b)
        else:
            with torch.cuda.stream(main_stream):
                y_a = model_a(x_a)
                y_b = model_b(x_b)

        end.record(main_stream)
        end.synchronize()  # CPU 等待终点，之后才读取时间。
        wall_ms = (perf_counter() - wall_start) * 1000
        gpu_ms = start.elapsed_time(end)
        return gpu_ms, wall_ms, (y_a, y_b)

    # 两种路径分别预热；预热真实走完计时与汇合路径，四个 Event 也随之完成初始化。
    # 每次 run 返回前均已完成目标工作，因此复用 Event 是安全的。
    for _ in range(5):
        run(False)
        run(True)

    samples = {False: [], True: []}
    for index in range(20):
        # 交替先后顺序，减轻固定顺序带来的偏差。
        order = (False, True) if index % 2 == 0 else (True, False)
        for mode in order:
            gpu_ms, wall_ms, outputs = run(mode)
            samples[mode].append((gpu_ms, wall_ms))

    serial_outputs = run(False)[2]
    multi_outputs = run(True)[2]
    for expected, actual in zip(serial_outputs, multi_outputs):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    print("GPU:", torch.cuda.get_device_name(device))
    print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
    for mode, label in [(False, "单 Stream"), (True, "双 Stream")]:
        gpu_ms = median(row[0] for row in samples[mode])
        wall_ms = median(row[1] for row in samples[mode])
        print(f"{label}: Event 区间 {gpu_ms:.3f} ms; 墙钟 {wall_ms:.3f} ms")
    print("两种调度的输出校验通过。")


if __name__ == "__main__":
    main()
