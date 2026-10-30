"""打开异步传输，为什么还是没变快？（第7章 · 访存与传输）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch。
"""
import sys
import time
from statistics import median
import torch

MODES = ("blocking", "same_stream", "overlap")


def make_runner(host_batches, weight, device):
    compute_stream = torch.cuda.current_stream(device)
    copy_stream = torch.cuda.Stream(device=device)
    inputs = [torch.empty_like(host_batches[0], device=device) for _ in range(2)]
    outputs = [
        torch.empty(
            (h.shape[0], weight.shape[1]), dtype=h.dtype, device=device
        ) for h in host_batches
    ]
    ready = [torch.cuda.Event() for _ in range(2)]
    done = [torch.cuda.Event() for _ in range(2)]

    def enqueue(mode):
        if mode not in MODES:
            raise ValueError(mode)
        # 调用者在每轮前后等待设备完成；轮内不全局同步。
        used = [False, False]
        with torch.cuda.stream(compute_stream):
            if mode == "overlap":
                # 包括初始化/缓存分配器相关的 stream 顺序要求。
                copy_stream.wait_stream(compute_stream)
            for batch_id, host_batch in enumerate(host_batches):
                slot = batch_id % 2
                if mode == "overlap":
                    with torch.cuda.stream(copy_stream):
                        if used[slot]:
                            copy_stream.wait_event(done[slot])
                        inputs[slot].copy_(host_batch, non_blocking=True)
                        ready[slot].record(copy_stream)
                    compute_stream.wait_event(ready[slot])
                else:
                    inputs[slot].copy_(
                        host_batch, non_blocking=(mode == "same_stream")
                    )

                torch.mm(inputs[slot], weight, out=outputs[batch_id])
                if mode == "overlap":
                    done[slot].record(compute_stream)
                    used[slot] = True

    return enqueue, outputs


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要可用的 CUDA GPU 和 CUDA 版 PyTorch。")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(device))

    count, rows, width = 12, 512, 1024
    host_batches = [
        torch.randn(rows, width, dtype=torch.float32, device="cpu", pin_memory=True)
        for _ in range(count)
    ]
    weight = torch.randn(width, width, device=device, dtype=torch.float32) * 0.01
    enqueue, outputs = make_runner(host_batches, weight, device)
    print(f"batches={count}, input=({rows}, {width}), FP32, two input slots")

    torch.cuda.synchronize(device)
    enqueue("blocking")
    torch.cuda.synchronize(device)
    reference = [out.clone() for out in outputs]
    torch.cuda.synchronize(device)

    for mode in MODES:
        # 先清成 NaN：若本轮漏写某一批，比较会失败，
        # 而不会被上一轮留下的正确旧输出掩盖。
        for out in outputs:
            out.fill_(float("nan"))
        torch.cuda.synchronize(device)
        enqueue(mode)
        torch.cuda.synchronize(device)
        for out, expected in zip(outputs, reference):
            torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
        print(mode, "all batch results match the blocking reference")

    timings = {mode: [] for mode in MODES}
    for round_id in range(6):
        shift = round_id % len(MODES)
        order = MODES[shift:] + MODES[:shift]
        for mode in order:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            enqueue(mode)
            torch.cuda.synchronize(device)
            timings[mode].append((time.perf_counter() - start) * 1000)
    for mode in MODES:
        samples = timings[mode]
        mid = median(samples)
        print(
            f"{mode}: round median={mid:.4f} ms"
            f", range=[{min(samples):.4f}, {max(samples):.4f}] ms"
            f", throughput={count / (mid / 1000):.2f} batches/s"
        )

    if "--trace" in sys.argv:
        for mode in MODES:
            torch.cuda.synchronize(device)
            with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]) as prof:
                enqueue(mode)
                torch.cuda.synchronize(device)
            path = f"{mode}_trace.json"
            prof.export_chrome_trace(path)
            print("trace:", path)


if __name__ == "__main__":
    main()
