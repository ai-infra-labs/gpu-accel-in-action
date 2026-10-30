"""GPU 各算各的，结果怎么对齐？讲透 NCCL 通信（第5章 · 并行与分布式）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch。
"""
import os

import torch
import torch.distributed as dist


def main():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 2 or not torch.cuda.is_available():
        raise SystemExit("请在单机两张 NVIDIA GPU 上，用 torchrun 启动两个进程。")
    if torch.cuda.device_count() < 2:
        raise SystemExit("本示例要求每个进程可见同样的两张 GPU。")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        values = [1, 2, 3, 4] if rank == 0 else [10, 20, 30, 40]
        source = torch.tensor(values, dtype=torch.float32, device=device)

        reduced = source.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

        raw_parts = [torch.empty_like(source) for _ in range(world)]
        dist.all_gather(raw_parts, source)
        expected_raw = torch.tensor(
            [1, 2, 3, 4, 10, 20, 30, 40],
            dtype=source.dtype,
            device=device,
        )
        torch.testing.assert_close(torch.cat(raw_parts), expected_raw)

        # 输入长度4，每个 rank 接收归约结果中的2个元素。
        shard = torch.empty(2, dtype=source.dtype, device=device)
        dist.reduce_scatter_tensor(shard, source, op=dist.ReduceOp.SUM)

        result_parts = [torch.empty_like(shard) for _ in range(world)]
        dist.all_gather(result_parts, shard)
        reconstructed = torch.cat(result_parts)

        expected = torch.tensor([11, 22, 33, 44],
                                dtype=source.dtype, device=device)
        torch.testing.assert_close(reduced, expected)
        torch.testing.assert_close(reconstructed, expected)
        expected_shard = expected[rank * 2:(rank + 1) * 2]
        torch.testing.assert_close(shard, expected_shard)

        # 示例显式等目标设备完成，再读取到 CPU 展示。
        torch.cuda.synchronize(device)
        if rank == 0:
            print("AllReduce:", reduced.cpu().tolist())
            print("原始输入 AllGather:", torch.cat(raw_parts).cpu().tolist())
            print("ReduceScatter 后再 AllGather:", reconstructed.cpu().tolist())
            print("结果校验通过；本例不测试通信速度。")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
