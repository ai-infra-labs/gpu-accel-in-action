"""删了张量，显存为什么还没降？（第7章 · 访存与传输）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch。
"""
import gc
import torch
from torch import nn


def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要可用的 CUDA GPU 和 CUDA 版 PyTorch。")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(42)

    print("PyTorch:", torch.__version__)
    print("PyTorch 构建使用的 CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(device))
    print("分配器后端:", torch.cuda.memory.get_allocator_backend())

    def report(stage):
        # 诊断时等待本设备工作完成，便于对照阶段；不是测速代码。
        torch.cuda.synchronize(device)
        mib = 1024 ** 2
        allocated = torch.cuda.memory_allocated(device) / mib
        reserved = torch.cuda.memory_reserved(device) / mib
        peak = torch.cuda.max_memory_allocated(device) / mib
        print(
            f"{stage:<18} | allocated={allocated:9.2f} MiB"
            f" | reserved={reserved:9.2f} MiB"
            f" | peak={peak:9.2f} MiB"
        )

    report("初始状态")
    model = nn.Sequential(
        nn.Linear(1000, 10000),
        nn.ReLU(),
        nn.Linear(10000, 1000),
    ).to(device=device, dtype=torch.float32)

    # 固定实现选项，减少 foreach 临时张量对教学观察的干扰。
    optimizer = torch.optim.Adam(model.parameters(), foreach=False)
    criterion = nn.MSELoss()
    parameter_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    print(f"参数理论存储量: {parameter_bytes / 1024 ** 2:.2f} MiB")
    report("模型与优化器创建后")

    for step in range(1, 3):
        print(f"\n第 {step} 轮")
        optimizer.zero_grad(set_to_none=True)
        # 重置峰值统计，不释放显存；从本轮起重新观察高水位。
        torch.cuda.reset_peak_memory_stats(device)
        inputs = torch.randn(64, 1000, device=device, dtype=torch.float32)
        target = torch.randn(64, 1000, device=device, dtype=torch.float32)
        report("输入创建后")

        output = model(inputs)
        loss = criterion(output, target)
        report("前向后")

        loss.backward()
        report("反向后")
        optimizer.step()
        report("参数更新后")

        del inputs, target, output, loss
        report("删除本轮临时对象后")
        optimizer.zero_grad(set_to_none=True)
        report("梯度设为 None 后")

        if step == 1:
            # 只作一次观察对照，不建议照搬为每轮训练的固定步骤。
            torch.cuda.empty_cache()
            report("清理空闲缓存后")

    # 实验结束：优化器也持有参数引用，所以与模型一起删除。
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    report("删除模型与优化器后")


if __name__ == "__main__":
    main()
