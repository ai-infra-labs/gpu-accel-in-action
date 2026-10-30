"""一层矩阵太大，怎样让多张 GPU 合力算？（第5章 · 并行与分布式）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch。
"""
import os

import torch
import torch.distributed as dist


@torch.inference_mode()
def main():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 2 or not torch.cuda.is_available():
        raise SystemExit("请在单机两张 NVIDIA GPU 上启动两个进程。")
    if torch.cuda.device_count() < 2:
        raise SystemExit("本示例要求各进程可见同样的两张 GPU。")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        x_cpu = torch.tensor([[1., 2., 3., 4.]])
        wa_cpu = torch.tensor([[1., 0., 2., 0.], [0., 1., 0., 2.],
                               [1., 1., 0., 0.], [0., 0., 1., 1.]])
        ba_cpu = torch.tensor([0., -6., 1., -10.])
        wb_cpu = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 2.]])
        bb_cpu = torch.tensor([1., -1.])
        start, stop = rank * 2, (rank + 1) * 2

        x = x_cpu.to(device)  # 同一个样本，在两个 TP rank 上可用。
        wa = wa_cpu[:, start:stop].contiguous().to(device)
        ba = ba_cpu[start:stop].contiguous().to(device)
        wb = wb_cpu[start:stop, :].contiguous().to(device)
        bb = bb_cpu.to(device)

        h_local = torch.relu(x @ wa + ba)
        partial = h_local @ wb  # 两层之间没有 AllGather。
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        output = partial + bb  # 最终偏置在求和后加一次，不除以卡数。

        expected_cpu = torch.relu(x_cpu @ wa_cpu + ba_cpu) @ wb_cpu + bb_cpu
        torch.testing.assert_close(output, expected_cpu.to(device))
        torch.cuda.synchronize(device)
        if rank == 0:
            print("TP 前向输出:", output.cpu().tolist())
            print("与完整网络输出一致；本例不测试速度。")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
