"""两张 GPU 传数据，真的走了 NVLink 吗？（第7章 · 访存与传输）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch。
"""
import time
from statistics import median
import torch

ROUTES = ("host_staged", "framework_copy")


def synchronize_pair(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def one_sample(route, source, target, staging, devices):
    synchronize_pair(devices)
    start = time.perf_counter()
    if route == "host_staged":
        with torch.cuda.device(source.device):
            staging.copy_(source, non_blocking=True)
        # CPU 缓冲区完成接收后，才允许目标 GPU 从中读取。
        torch.cuda.synchronize(source.device)
        with torch.cuda.device(target.device):
            target.copy_(staging, non_blocking=True)
    elif route == "framework_copy":
        with torch.cuda.device(target.device):
            target.copy_(source, non_blocking=True)
    else:
        raise ValueError(route)
    synchronize_pair(devices)
    return (time.perf_counter() - start) * 1000


def run_case(source_id, target_id, payload_bytes):
    source_device = torch.device(f"cuda:{source_id}")
    target_device = torch.device(f"cuda:{target_id}")
    devices = (source_device, target_device)
    elements = payload_bytes // 4
    source = torch.randn(elements, dtype=torch.float32, device=source_device)
    target = torch.empty(elements, dtype=torch.float32, device=target_device)
    staging = torch.empty(
        elements, dtype=torch.float32, device="cpu", pin_memory=True
    )
    synchronize_pair(devices)
    reference = source.cpu()  # 正确性参照，放在计时外。
    print(f"\n{source_id} -> {target_id}, payload={payload_bytes} bytes")

    for route in ROUTES:
        for _ in range(3):
            one_sample(route, source, target, staging, devices)
        # 清成 NaN 后再完整执行一次：若本路线漏写目标，
        # 比较会失败，而不会被之前路线留下的正确旧结果掩盖。
        target.fill_(float("nan"))
        one_sample(route, source, target, staging, devices)
        torch.testing.assert_close(target.cpu(), reference, rtol=0, atol=0)
        print(f"  {route}: correctness PASS")

    records = {route: [] for route in ROUTES}
    for round_id in range(8):
        order = ROUTES if round_id % 2 == 0 else ROUTES[::-1]
        for route in order:
            records[route].append(
                one_sample(route, source, target, staging, devices)
            )
    for route in ROUTES:
        samples = records[route]
        mid = median(samples)
        if mid <= 0:
            raise RuntimeError("计时分辨率不足，无法计算有效速率。")
        gbps = payload_bytes / (mid / 1000) / 1e9
        print(
            f"  {route}: completion median={mid:.6f} ms"
            f", range=[{min(samples):.6f}, {max(samples):.6f}] ms"
            f", payload delivery={gbps:.3f} GB/s"
        )


@torch.no_grad()
def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("需要至少两张可见的 CUDA GPU。")
    torch.manual_seed(42)
    print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
    for device_id in (0, 1):
        props = torch.cuda.get_device_properties(device_id)
        print(
            "logical device:", device_id, "name:", props.name,
            "UUID:", getattr(props, "uuid", "本版本未提供此属性")
        )
    print("device 0 can access device 1:", torch.cuda.can_device_access_peer(0, 1))
    print("device 1 can access device 0:", torch.cuda.can_device_access_peer(1, 0))
    print("warmup=3 per route, samples=8 per route; physical route unconfirmed")
    for source_id, target_id in ((0, 1), (1, 0)):
        for payload_bytes in (1 << 10, 1 << 20, 16 << 20, 64 << 20):
            run_case(source_id, target_id, payload_bytes)


if __name__ == "__main__":
    main()
