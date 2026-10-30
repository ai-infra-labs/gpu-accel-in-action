"""在模型运行之后，检查原空间输出，并计算单病例前景Dice。"""
import argparse
import json
import nibabel as nib
import numpy as np


def check_prediction(image_path, label_path, pred_path):
    image, label, pred = [nib.load(path) for path in (image_path, label_path, pred_path)]
    for name, volume in [("image", image), ("label", label), ("prediction", pred)]:
        if len(volume.shape) != 3 or volume.header.get_xyzt_units()[0] != "mm":
            raise ValueError(f"{name}必须是空间单位为mm的三维体积。")
        if not np.isfinite(volume.affine).all():
            raise ValueError(f"{name}包含无效affine。")
    for volume in (label, pred):
        if volume.shape != image.shape or not np.allclose(
            volume.affine, image.affine, rtol=0, atol=1e-3
        ):
            raise ValueError("影像、标签、预测不在同一原始网格，停止逐体素比较。")
    truth, result = [volume.get_fdata(dtype=np.float32) for volume in (label, pred)]
    for values in (truth, result):
        if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
            raise ValueError("标签与预测必须是有限数值的0/1掩码。")
    truth, result = truth == 1, result == 1
    denominator = int(truth.sum()) + int(result.sum())
    dice = 2.0 * int(np.logical_and(truth, result).sum()) / denominator if denominator else None
    # 双方均空时记为None，不将其悄悄计成完美分割。
    return {"foreground_dice": dice, "reference_voxels": int(truth.sum()),
            "prediction_voxels": int(result.sum()), "same_grid": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("label")
    parser.add_argument("prediction")
    args = parser.parse_args()
    print(json.dumps(check_prediction(args.image, args.label, args.prediction), indent=2))
