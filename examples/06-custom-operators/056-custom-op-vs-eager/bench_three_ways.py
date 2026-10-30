"""手写 CUDA 一定更快吗？三种实现同场比较（第6章 · 自定义算子）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch。
"""
import os
import random
import statistics
import time
import torch
from affine import affine_relu

def eager(x):
    return torch.relu(x * 2.0 - 1.0)

def fused(x):
    return affine_relu(x, 2.0, -1.0)

def measure_group_ms(fn, x, repeats=100):
    torch.cuda.synchronize(x.device)
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn(x)
    torch.cuda.synchronize(x.device)
    return (time.perf_counter() - start) * 1000.0 / repeats

def check_close(fn, x):
    torch.testing.assert_close(fn(x), eager(x), rtol=1e-5, atol=1e-6)

def profile_once(functions, x):
    from torch.profiler import profile, record_function, ProfilerActivity
    torch.cuda.synchronize(x.device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for name, fn in functions.items():
            with record_function(name):
                result = fn(x)
        torch.cuda.synchronize(x.device)
    prof.export_chrome_trace("affine_relu_trace.json")
    print("Trace written: affine_relu_trace.json")

@torch.inference_mode()
def main():
    torch.manual_seed(0)
    order_rng = random.Random(0)
    print("PyTorch:", torch.__version__,
          "PyTorch built with CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("dtype=float32; scale=2; bias=-1; repeats=100; rounds=7")
    print("compile: backend=inductor, mode=default, fullgraph=True, dynamic=False")

    for n in (0, 1, 255, 256, 257):
        check_close(fused, torch.rand(n, device="cuda") * 4.0 - 2.0)
    special = torch.tensor([-2., 0., 0.5, 1., 2.], device="cuda")
    check_close(fused, special)

    compiled = None
    try:
        compiled = torch.compile(eager, backend="inductor", mode="default",
                                 fullgraph=True, dynamic=False)
    except Exception as error:
        print("UNAVAILABLE compiled:", type(error).__name__, str(error)[:200])

    for n in (1_024, 65_536, 1_000_000, 8_000_000):
        x = torch.rand(n, device="cuda", dtype=torch.float32) * 4.0 - 2.0
        functions = {"eager": eager, "fused": fused}
        check_close(fused, x)
        if compiled is not None:
            try:
                actual = compiled(x)  # 首次编译或该形状的重编译在计时外。
            except Exception as error:
                print("UNAVAILABLE compiled for N=", n,
                      type(error).__name__, str(error)[:200])
            else:
                # 正确性失败应直接报错，不能当成编译不可用而略过。
                torch.testing.assert_close(actual, eager(x), rtol=1e-5, atol=1e-6)
                functions["compiled"] = compiled

        for fn in functions.values():
            for _ in range(20):
                fn(x)
        torch.cuda.synchronize(x.device)
        samples = {name: [] for name in functions}
        for _ in range(7):
            order = list(functions)
            order_rng.shuffle(order)
            for name in order:
                samples[name].append(measure_group_ms(functions[name], x))

        baseline = statistics.median(samples["eager"])
        print(f"\nN={n}; input={x.numel() * x.element_size() / 2**20:.3f} MiB")
        for name, values in samples.items():
            median = statistics.median(values)
            p25, _, p75 = statistics.quantiles(values, n=4, method="inclusive")
            print(f"{name:8s} median={median:.6f} ms "
                  f"p25={p25:.6f} p75={p75:.6f} "
                  f"eager/time={baseline / median:.3f}x")

    if os.environ.get("PROFILE") == "1":
        profile_once(functions, x)  # 最后一个规模，且仅记录可用路径。

if __name__ == "__main__":
    main()
