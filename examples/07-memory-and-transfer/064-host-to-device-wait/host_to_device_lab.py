"""数据还没送到，GPU 已经等急了（第7章 · 访存与传输）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch。
"""
import time
from statistics import median
import torch

CASES = ("pageable", "pinned_ready", "pin_then_copy")


def one_sample(kind, pageable, pinned, dst, device):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    if kind == "pageable":
        src = pageable
    elif kind == "pinned_ready":
        src = pinned
    elif kind == "pin_then_copy":
        src = pageable.pin_memory()
    else:
        raise ValueError(kind)
    prepared = time.perf_counter()

    dst.copy_(src, non_blocking=True)
    submitted = time.perf_counter()
    # src 保留到复制完成，避免本实验提前复用主机缓冲区。
    torch.cuda.synchronize(device)
    finished = time.perf_counter()
    return {
        "prepare_ms": (prepared - start) * 1000,
        "call_ms": (submitted - prepared) * 1000,
        "complete_ms": (finished - start) * 1000,
    }


def run_case(elements, device):
    pageable = torch.randn(elements, dtype=torch.float32, device="cpu")
    pinned = pageable.pin_memory()  # B 的准备成本特意放在计时外。
    dst = torch.empty(elements, dtype=torch.float32, device=device)
    assert not pageable.is_pinned()
    assert pinned.is_pinned()
    size_mib = pageable.numel() * pageable.element_size() / 1024 ** 2
    print(f"\nFP32 payload: {size_mib:.2f} MiB")

    for kind in CASES:
        for _ in range(2):
            one_sample(kind, pageable, pinned, dst, device)
        # 正确性检查在计时外，回传结果与源数据逐元素比较。
        torch.testing.assert_close(dst.cpu(), pageable, rtol=0, atol=0)
        print(f"  {kind}: correctness PASS")

    records = {kind: [] for kind in CASES}
    for round_id in range(9):
        shift = round_id % len(CASES)
        order = CASES[shift:] + CASES[:shift]
        for kind in order:
            records[kind].append(
                one_sample(kind, pageable, pinned, dst, device)
            )

    for kind in CASES:
        samples = records[kind]
        totals = [s["complete_ms"] for s in samples]
        preparation = median(s["prepare_ms"] for s in samples)
        call = median(s["call_ms"] for s in samples)
        print(
            f"  {kind}: completion median={median(totals):.4f} ms"
            f", range=[{min(totals):.4f}, {max(totals):.4f}] ms"
            f"; prepare median={preparation:.4f} ms"
            f", copy call median={call:.4f} ms"
        )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要可用的 CUDA GPU 和 CUDA 版 PyTorch。")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.manual_seed(42)
    print("PyTorch:", torch.__version__)
    print("PyTorch 构建使用的 CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(device))
    print("default stream, preallocated GPU destination, non_blocking=True")
    print("warmup=2 per path, samples=9 per path")
    for elements in (1 << 18, 1 << 22, 1 << 24):
        run_case(elements, device)


if __name__ == "__main__":
    main()
