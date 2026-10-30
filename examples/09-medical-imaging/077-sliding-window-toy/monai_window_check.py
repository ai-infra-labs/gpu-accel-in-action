"""检查MONAI滑窗的张量与设备连接；假预测器不用于器官分割。"""
import argparse
import torch
from monai.inferers import sliding_window_inference


class Pointwise(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.tensor(0.0))

    def forward(self, patch):
        # 核对窗口确实进入了模型所在设备。
        if patch.device != self.anchor.device:
            raise RuntimeError("窗口与预测器不在同一设备。")
        return torch.cat([-patch, patch], dim=1)  # 两通道分数，不是学习到的结果。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前PyTorch未检测到可用设备。")
    model = Pointwise().to(args.device).eval()
    generator = torch.Generator().manual_seed(2026)
    # 整个输入留在CPU；末轴95小于窗口96，会触发补齐与裁回。
    volume = torch.randn((1, 1, 111, 128, 95), generator=generator, dtype=torch.float32)
    expected = torch.cat([-volume, volume], dim=1)
    with torch.inference_mode():
        for mode in ("constant", "gaussian"):
            for batch in (1, 3):
                output = sliding_window_inference(
                    inputs=volume, roi_size=(96, 96, 96), sw_batch_size=batch,
                    predictor=model, overlap=0.5, mode=mode, sigma_scale=0.125,
                    padding_mode="constant", cval=0.0,
                    sw_device=args.device, device="cpu", buffer_steps=None,
                )  # GPU模式下仅窗口与预测器在GPU，融合结果留在CPU。
                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
                assert output.device.type == "cpu" and output.shape == expected.shape
                print("PASS", mode, "sw_batch_size", batch, "shape", tuple(output.shape))
    print("检查的是滑窗连接，不能证明真实网络的分割质量或GPU性能。")


if __name__ == "__main__":
    main()
