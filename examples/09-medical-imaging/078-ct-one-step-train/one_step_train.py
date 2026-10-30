"""只完成一次训练更新；默认合成数据，可选一对真实NIfTI，不能生成可用器官模型。"""
import argparse
import numpy as np
import torch
from monai.data import list_data_collate
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.optimizers import Novograd
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, SpatialPadd, RandCropByPosNegLabeld,
)


def prepare_real(image_path, label_path):
    import nibabel as nib
    image, label = nib.load(image_path), nib.load(label_path)
    for volume in (image, label):
        if len(volume.shape) != 3 or volume.header.get_xyzt_units()[0] != "mm":
            raise ValueError("要求空间单位为mm的三维NIfTI。")
        if not np.isfinite(volume.affine).all():
            raise ValueError("存在无效空间变换。")
    if image.shape != label.shape or not np.allclose(
        image.affine, label.affine, rtol=0, atol=1e-3
    ):
        raise ValueError("原始影像与标签不在同一网格，先停止训练。")
    # 标签在高精度下校验：FP32舍入可能把1.00000001变成1.0后蒙混过关。
    values = label.get_fdata(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("本例只接受0/1二值标签。")
    if not np.isfinite(image.get_fdata(dtype=np.float32)).all():
        raise ValueError("影像含非有限值。")
    transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # 连续影像插值与离散标签插值分开；两者使用相同目标空间。
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=-57, a_max=164,
                             b_min=0, b_max=1, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])
    return transform({"image": image_path, "label": label_path})


def synthetic_case():
    coords = np.indices((48, 48, 48))
    sphere = ((coords - 24) ** 2).sum(axis=0) < 10 ** 2
    rng = np.random.default_rng(2026)
    # 人为让球内强度更高，只为检查张量和梯度；不模拟真实脾脏。
    image = 0.1 + 0.2 * rng.random(sphere.shape) + 0.5 * sphere
    return {"image": image[None].astype(np.float32),
            "label": sphere[None].astype(np.int64)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image")
    parser.add_argument("--label")
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--roi", type=int, default=32)
    args = parser.parse_args()
    if bool(args.image) != bool(args.label):
        parser.error("image和label必须同时提供。")
    if args.roi < 8 or args.roi % 4:
        parser.error("教学小网络要求roi为不小于8的4的倍数。")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前PyTorch没有可用CUDA设备。")
    torch.manual_seed(2026)  # 便于重复检查；不承诺跨设备逐位一致。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False  # 明确本例不采用TF32作为FP32卷积近似。
    case = prepare_real(args.image, args.label) if args.image else synthetic_case()
    roi = (args.roi,) * 3
    sample = Compose([
        EnsureTyped(keys=["image", "label"]),
        SpatialPadd(keys=["image", "label"], spatial_size=roi,
                    mode="constant"),  # 教学适配：避免短轴导致随机裁剪失败。
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=roi,
            pos=1, neg=1, num_samples=2, image_key="image", image_threshold=0,
        ),
    ])
    sample.set_random_state(seed=2026)
    batch = list_data_collate(sample(case))  # 两个小块合成窗口批次，不是两个患者。
    x = batch["image"].to(args.device, dtype=torch.float32)
    target = batch["label"].to(args.device, dtype=torch.long)
    assert x.shape == target.shape and x.shape[0] == 2
    # 缩小的随机网络，只检查训练链路；与官方预训练网络结构不同。
    model = UNet(spatial_dims=3, in_channels=1, out_channels=2,
                 channels=(4, 8, 16), strides=(2, 2),
                 num_res_units=1, norm="batch").to(args.device)
    criterion = DiceCELoss(
        include_background=True, to_onehot_y=True, softmax=True,
        squared_pred=True, batch=True, smooth_nr=1e-5, smooth_dr=1e-5,
        lambda_dice=0.5, lambda_ce=0.5,
    )  # 保留所选官方配置的损失定义。
    optimizer = Novograd(model.parameters(), lr=0.002)
    model.train()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.zero_grad(set_to_none=True)  # 本次只累积这一批的梯度。
    logits = model(x)  # 标签没有进入模型输入，仍用于后面的监督比较。
    assert logits.shape == (2, 2) + roi
    loss = criterion(logits, target)  # 不先softmax，也不先argmax。
    if not torch.isfinite(loss):
        raise RuntimeError("损失为非有限值。")
    loss.backward()  # 根据损失计算参数梯度，尚未更新参数。
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all().item() for g in grads):
        raise RuntimeError("缺少梯度或存在非有限梯度。")
    if not any(torch.count_nonzero(g).item() > 0 for g in grads):
        raise RuntimeError("所有梯度为0，请检查训练链路。")
    optimizer.step()  # 优化器根据梯度及其内部状态更新权重。
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError("更新后出现非有限参数。")
    delta = max((p.detach() - old).abs().max().item()
                for p, old in zip(model.parameters(), before))
    if delta == 0:
        raise RuntimeError("参数没有发生变化。")
    print("输入/标签/logits形状", tuple(x.shape), tuple(target.shape), tuple(logits.shape))
    print("本批更新前loss", loss.item(), "最大参数变化", delta)
    print("只完成一次更新：不报告验证Dice，不保存或宣称可用器官模型。")


if __name__ == "__main__":
    main()
