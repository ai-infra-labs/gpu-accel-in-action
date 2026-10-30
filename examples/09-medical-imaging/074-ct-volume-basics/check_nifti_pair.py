"""教学用：检查同网格的二分类NIfTI影像/标签，生成三平面叠加图。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无桌面环境也可以保存图片。
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np


def load_volume(path):
    volume = nib.load(path)  # 读取文件头及数据访问对象。
    if len(volume.shape) != 3:
        raise ValueError("本例只处理三维体积，不自动压缩四维数据。")
    if volume.header.get_xyzt_units()[0] != "mm":
        raise ValueError("本例要求空间单位明确为mm，请先核实单位。")
    affine = volume.affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError("空间变换无效，请检查文件来源。")
    qform, qcode = volume.get_qform(coded=True)
    sform, scode = volume.get_sform(coded=True)
    if qcode and scode and not np.allclose(qform, sform, rtol=0, atol=1e-3):
        raise ValueError("qform与sform不一致，需人工确认应采用的空间。")
    return volume


def inspect_pair(image_path, label_path, output_path):
    image = load_volume(image_path)
    label = load_volume(label_path)
    # 形状和空间映射必须同时匹配；不通过修改文件头伪造对齐。
    if image.shape != label.shape or not np.allclose(
        image.affine, label.affine, rtol=0, atol=1e-3
    ):
        raise ValueError("图像与标签不在本例要求的同一网格上。")

    original = {"shape": list(image.shape), "affine": image.affine.tolist()}
    # 重排/翻转数组，同时更新affine；不插值，不统一体素间距。
    # 拒绝仍有斜切/剪切的情况，避免把斜切面标成标准解剖平面。
    image = nib.as_closest_canonical(image, enforce_diag=True)
    label = nib.as_closest_canonical(label, enforce_diag=True)
    if image.shape != label.shape or not np.allclose(
        image.affine, label.affine, rtol=0, atol=1e-3
    ):
        raise ValueError("重排后的图像与标签网格不一致。")
    if nib.aff2axcodes(image.affine) != ("R", "A", "S"):
        raise ValueError("本例只绘制轴对齐的RAS数据。")

    data = image.get_fdata(dtype=np.float32)  # 应用文件数值缩放，明确使用FP32。
    # 标签用更高精度读取后再检查二值：FP32转换本身也会做舍入，
    # 例如1.00000001会先变成1.0，从而掩盖非法值。
    mask = label.get_fdata(dtype=np.float64)
    if not np.isfinite(data).all() or not np.isfinite(mask).all():
        raise ValueError("图像或标签包含NaN/Inf。")
    if not np.isin(mask, [0, 1]).all() or not np.any(mask == 1):
        raise ValueError("本例要求非空二分类标签：0为背景，1为目标。")
    mask = mask.astype(bool)  # 完成有限值与二值检查后，再转为布尔数组。
    foreground = mask
    spacing = nib.affines.voxel_sizes(image.affine)
    # 每个方向独立选择前景面积最大的切片，只用于展示参考标注。
    indices = [int(np.argmax(foreground.sum(axis=tuple(j for j in range(3) if j != a))))
               for a in range(3)]
    nx, ny, nz = data.shape
    sx, sy, sz = spacing
    i, j, k = indices
    views = [
        ("Axial", data[:, :, k].T, foreground[:, :, k].T, nx, ny, sx, sy, "L -> R", "P -> A", k),
        ("Coronal", data[:, j, :].T, foreground[:, j, :].T, nx, nz, sx, sz, "L -> R", "I -> S", j),
        ("Sagittal", data[i, :, :].T, foreground[i, :, :].T, ny, nz, sy, sz, "P -> A", "I -> S", i),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained")
    for ax, (name, section, region, width, height, dx, dy, xdir, ydir, index) in zip(axes, views):
        # 坐标为切片内部相对毫米；半体素边界让像素中心落在0、dx、2dx……。
        extent = (-dx / 2, (width - 0.5) * dx, -dy / 2, (height - 0.5) * dy)
        ax.imshow(section, cmap="gray", origin="lower", extent=extent,
                  aspect="equal", vmin=-100, vmax=200, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~region, region), cmap=ListedColormap(["lime"]),
                  origin="lower", extent=extent, aspect="equal", alpha=0.35,
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set(title=f"{name}: index {index}", xlabel=f"{xdir} (relative mm)",
               ylabel=f"{ydir} (relative mm)")
    fig.suptitle("Reference mask overlay | teaching view, not a prediction")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    report = {"original": original, "display_shape": list(data.shape),
              "display_affine": image.affine.tolist(), "axes": list(nib.aff2axcodes(image.affine)),
              "spacing_mm": spacing.tolist(), "dtype": str(data.dtype),
              "array_MiB": data.nbytes / 2**20, "selected_ijk": indices}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {output}; metadata: {output.with_suffix('.json')}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="CT的NIfTI路径")
    parser.add_argument("label", help="同病例参考标签的NIfTI路径")
    parser.add_argument("--out", default="ct_three_planes.png")
    args = parser.parse_args()
    inspect_pair(args.image, args.label, args.out)
