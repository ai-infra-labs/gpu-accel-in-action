"""单GPU推理教学：比较同流串行和双缓冲复制/计算，不读取真实CT。"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn


class Runner:
    def __init__(self, model, host_batches):
        self.model, self.host = model, host_batches
        if not self.host or not all(x.device.type == "cpu" and x.is_pinned()
                                    and x.dtype == torch.float32 for x in self.host):
            raise ValueError("要求非空、锁页、FP32主机批次列表。")
        if not all(x.shape == self.host[0].shape for x in self.host):
            raise ValueError("本例只处理同尺寸批次。")
        # 对象在所有运行结束并同步之前始终存活，固定主机数据也不再修改。
        self.slots = [torch.empty_like(self.host[0], device="cuda") for _ in range(2)]
        self.compute = torch.cuda.current_stream()
        self.copy = torch.cuda.Stream()
        self.ready = [torch.cuda.Event() for _ in self.host]
        self.done = [torch.cuda.Event() for _ in self.host]

    @torch.inference_mode()
    def run(self, pipelined, collect=False):
        outputs = []
        if not pipelined:
            with torch.cuda.stream(self.compute):
                for host in self.host:
                    self.slots[0].copy_(host, non_blocking=True)
                    out = self.model(self.slots[0])  # 同流：本批复制排在本批计算之前。
                    if collect:
                        outputs.append(out)
                    del out
            return outputs

        # 槽位在主流上分配；先建立初始化到复制流的依赖。
        self.copy.wait_stream(self.compute)

        def stage(i):
            with torch.cuda.stream(self.copy):
                if i >= 2:
                    self.copy.wait_event(self.done[i - 2])  # 槽位上次被读完才可覆盖。
                self.slots[i % 2].copy_(self.host[i], non_blocking=True)
                self.ready[i].record(self.copy)  # 记录这一批复制完成的点。

        stage(0)  # 预填首个槽位。
        for i in range(len(self.host)):
            with torch.cuda.stream(self.compute):
                self.compute.wait_event(self.ready[i])  # 这一批复制好，计算才可读。
                out = self.model(self.slots[i % 2])
                self.done[i].record(self.compute)  # 此推理网络已不再读取当前输入槽位。
                if collect:
                    outputs.append(out)
                del out
            if i + 1 < len(self.host):
                stage(i + 1)  # 可与已经提交的第i批计算重叠；不保证实际重叠。
        return outputs


def measure(runner, pipelined, repeats):
    for _ in range(3):
        runner.run(pipelined)
        torch.cuda.synchronize()  # 只在整轮边界同步，不在每批间强制同步。
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        runner.run(pipelined)
        torch.cuda.synchronize()  # 包括两条流中本轮工作的完成与排空。
        samples.append((time.perf_counter() - start) * 1000)
    return np.percentile(samples, [50, 10, 90]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", type=int, default=32)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--out", default="copy_compute_report.json")
    args = parser.parse_args()
    if min(args.roi, args.batches, args.repeats) < 1:
        parser.error("roi、batches、repeats必须为正整数。")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要CUDA；CPU调度例子请运行pipeline_schedule.py。")
    torch.cuda.set_device(0)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    shape = (2, 1, args.roi, args.roi, args.roi)  # 每批两个合成三维窗口。
    # 刻意在计时外准备小规模锁页列表，隔离H2D与计算；实际系统不应缓存所有CT。
    host = [torch.rand(shape).pin_memory() for _ in range(args.batches)]
    model = nn.Sequential(nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(),
                          nn.Conv3d(8, 2, 3, padding=1)).cuda().eval()
    runner = Runner(model, host)
    try:
        serial = runner.run(False, collect=True)
        torch.cuda.synchronize()
        reference = [x.cpu() for x in serial]
        del serial  # 正确性检查输出不计入后面的计时。
        parallel = runner.run(True, collect=True)
        torch.cuda.synchronize()
        assert len(parallel) == len(reference)
        for actual, expected in zip(parallel, reference):
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
        del parallel, reference
        serial_ms = measure(runner, False, args.repeats)
        pipeline_ms = measure(runner, True, args.repeats)
        report = {"torch": torch.__version__, "numpy": np.__version__,
                  "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "shape_per_batch": list(shape), "batches": args.batches,
                  "tf32": False, "cudnn_deterministic": True,
                  "warmup_rounds": 3, "repeats": args.repeats,
                  "serial_ms_median_p10_p90": serial_ms,
                  "pipeline_ms_median_p10_p90": pipeline_ms,
                  "synthetic_output_check": "passed"}
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.trace:
            from torch.profiler import profile, ProfilerActivity
            for label, mode in [("serial", False), ("pipeline", True)]:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    runner.run(mode)
                    torch.cuda.synchronize()
                trace = path.with_name(path.stem + "_" + label + "_trace.json")
                prof.export_chrome_trace(str(trace))  # 跟踪开销不纳入上面的计时。
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print("仅比较合成窗口的H2D+推理；没有真实CT、训练或全流程速度结论。")
    finally:
        # 所有流的工作完成后，才释放固定GPU槽位与锁页主机源。
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
