"""合成三维小块的AMP训练检查与同权重推理探针；不训练可用器官模型。"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import monai
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.optimizers import Novograd


def forward(model, x, dtype):
    # 参数和x保留FP32，由autocast选择有资格算子的计算类型。
    with torch.autocast("cuda", dtype=dtype or torch.float16, enabled=dtype is not None):
        logits = model(x)
    return logits.float()  # 明确输出边界；这不会恢复前面已经丢掉的信息。


def train_check(model, x, target, dtype, steps):
    optimizer = Novograd(model.parameters(), lr=0.002)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    criterion = DiceCELoss(
        include_background=True, to_onehot_y=True, softmax=True,
        squared_pred=True, batch=True, smooth_nr=1e-5, smooth_dr=1e-5,
        lambda_dice=0.5, lambda_ce=0.5,
    )
    model.train()
    rows = []
    skipped_nonfinite = 0
    real_step = optimizer.step
    update_calls = 0

    def counted_step(*args, **kwargs):
        # 包装一层，精确统计scaler是否真正调用了optimizer.step()。
        nonlocal update_calls
        update_calls += 1
        return real_step(*args, **kwargs)

    optimizer.step = counted_step
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = forward(model, x, dtype)
        with torch.autocast("cuda", enabled=False):
            loss = criterion(logits, target)  # Dice与CE在明确的FP32区域计算。
        if not torch.isfinite(loss).item():
            raise RuntimeError("前向损失非有限；先定位数值问题，不能靠缩放掩盖。")
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()  # 在前向autocast区域外启动反向。
        scaler.unscale_(optimizer)  # 先恢复梯度尺度，再检查；若剪裁也放在之后。
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads:
            raise RuntimeError("没有参数梯度。")
        finite = all(torch.isfinite(g).all().item() for g in grads)
        if not finite and not scaler.is_enabled():
            raise RuntimeError("未启用缩放的路径出现非有限梯度。")
        calls_before = update_calls
        scaler.step(optimizer)  # FP16路径发现非有限梯度时跳过优化器更新。
        scaler.update()  # 调整下次缩放；不要每个step重新创建scaler。
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError("参数已出现非有限值。")
        if not finite:
            skipped_nonfinite += 1
        rows.append({"step": step, "loss": loss.item(), "scale_before": scale_before,
                     "scale_after": scaler.get_scale(),
                     "update_skipped": update_calls == calls_before})
    summary = {"attempted_steps": steps,
               "skipped_nonfinite": skipped_nonfinite,
               "skipped_other": steps - skipped_nonfinite - update_calls,  # 如缩放因子异常导致的跳过。
               "optimizer_steps": update_calls,
               "effective_update_check": update_calls > 0}
    return rows, summary  # 局部优化器、梯度检查用引用在函数结束后释放。


def compare_same_weights(model, x, dtype):
    model.eval()
    with torch.inference_mode():
        reference = forward(model, x, None)
        candidate = forward(model, x, dtype)
        if not torch.isfinite(candidate).all().item() or not torch.isfinite(reference).all().item():
            raise RuntimeError("推理探针出现非有限输出。")
        error = (reference - candidate).abs()
        # 这里只比较两条执行路径，不计算器官分割的真值Dice。
        return {"max_abs_logit_error": error.max().item(),
                "mean_abs_logit_error": error.mean().item(),
                "label_disagreement_fraction":
                    (reference.argmax(1) != candidate.argmax(1)).float().mean().item()}


def measure_forward(model, x, dtype, repeats):
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            out = forward(model, x, dtype)
            del out
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = forward(model, x, dtype)
            torch.cuda.synchronize()  # 时间包括这次前向与输出转FP32的完成。
            samples.append((time.perf_counter() - start) * 1000)
            del out
        peak = torch.cuda.max_memory_allocated()
    return {"wall_ms_median_p10_p90": np.percentile(samples, [50, 10, 90]).tolist(),
            "peak_allocated_MiB": peak / 2 ** 20,
            "peak_above_baseline_MiB": (peak - base) / 2 ** 20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--roi", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out", default="amp_report.json")
    args = parser.parse_args()
    if args.roi < 8 or args.roi % 4 or args.steps < 1 or args.repeats < 1:
        parser.error("roi须为不小于8的4的倍数，steps与repeats须为正整数。")
    if not torch.cuda.is_available():
        raise RuntimeError("本脚本需要CUDA；CPU格式演示请运行fp16_numbers.py。")
    torch.cuda.set_device(0)
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("此PyTorch/CUDA设备组合未报告原生BF16支持。")
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False  # 固定FP32对照的TF32设置。
    torch.backends.cudnn.benchmark = False
    r = args.roi
    coords = torch.stack(torch.meshgrid(*[torch.arange(r)] * 3, indexing="ij"))
    sphere = ((coords - r // 2).float().square().sum(0) < (r / 4) ** 2)
    target = sphere.long()[None, None].repeat(2, 1, 1, 1, 1).cuda()
    noise = torch.rand((2, 1, r, r, r), device="cpu").cuda()
    x = (0.1 + 0.2 * noise + 0.5 * target.float()).float()
    # 与第5篇同类的缩小随机网络；不能加载官方脾脏模型检查点。
    model = UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(4, 8, 16),
                 strides=(2, 2), num_res_units=1, norm="batch").cuda().float()
    rows, update_summary = train_check(model, x, target, dtype, args.steps)
    model.zero_grad(set_to_none=True)  # 推理测量不保留训练参数梯度。
    comparison = compare_same_weights(model, x, dtype)
    report = {"torch": torch.__version__, "monai": monai.__version__,
              "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "precision": args.precision,
              "input_shape": list(x.shape), "tf32": False,
              "training_summary": update_summary, "training_checks": rows,
              "same_weights_inference": comparison}
    if args.benchmark:
        # 同一组训练后权重、同一输入，测两条推理路径；不是两种训练速度比较。
        report["inference_benchmark"] = {
            "scope": "device-resident forward plus output float32 cast",
            "warmup": 10, "repeats": args.repeats,
            "fp32_reference": measure_forward(model, x, None, args.repeats),
            "candidate": measure_forward(model, x, dtype, args.repeats),
        }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("有效更新检查:", update_summary["effective_update_check"],
          "实际优化器更新次数:", update_summary["optimizer_steps"])
    if not update_summary["effective_update_check"]:
        print("警告：所有训练步骤都被跳过，尚未完成有效更新检查。")
    print(json.dumps(report, indent=2))
    print("这是合成小块的训练连接与推理探针，不是CT分割质量或速度结论。")


if __name__ == "__main__":
    main()
