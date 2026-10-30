"""检查原始 CT 网格上的二值分割；不执行推理、不测量 GPU 性能。"""
import argparse  # 解析命令行中的清单路径和报告路径。
import csv  # 读取逐病例清单。
import hashlib  # 对输入文件计算摘要，帮助追踪具体版本。
import json  # 保存结构化结果，并禁止 NaN 混入报告。
import platform  # 记录实际 Python 版本。
from pathlib import Path  # 处理相对于清单目录的文件路径。
import nibabel as nib  # 读取 NIfTI 的数组与空间定义。
import numpy as np  # 计算集合计数、矩阵与分位数。
import scipy  # 记录 SciPy 版本。
from scipy import ndimage as ndi  # 提取边界并计算欧氏距离。


def sha256(path):
    h = hashlib.sha256()  # 初始化文件摘要，摘要不包含患者身份判断。
    with open(path, "rb") as stream:  # 用二进制方式读取文件。
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)  # 分块处理，避免一次读入整份压缩文件。
    return h.hexdigest()  # 返回可写入报告的十六进制摘要。


def load_geometry(path):
    image = nib.load(path)  # 读取头信息与数据代理，暂不加载整个 CT 数组。
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise ValueError("仅接受 NIfTI 图像")  # 限定本教学检查器的输入范围。
    if len(image.shape) != 3 or min(image.shape) <= 0:
        raise ValueError("必须是非空三维数组")  # 拒绝时间序列或空体积。
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("空间单位必须明确为 mm")  # 不猜测缺失单位。
    affine = image.affine  # 取 NiBabel 按头信息规则选择的体素到世界坐标变换。
    if not np.isfinite(affine).all():
        raise ValueError("affine 含非有限值")  # 阻止无效坐标进入距离计算。
    q, qc = image.get_qform(coded=True)  # 读取可选的 qform 与有效代码。
    s, sc = image.get_sform(coded=True)  # 读取可选的 sform 与有效代码。
    if qc and sc and not np.allclose(q, s, rtol=0, atol=1e-3):
        raise ValueError("qform/sform 不一致，空间定义待确认")  # 不自动选择冲突语义。
    axes = affine[:3, :3]  # 三列分别表示三个数组轴在物理空间中的步长向量。
    spacing = np.linalg.norm(axes, axis=0)  # 得到按数组轴顺序排列的毫米间距。
    if np.any(spacing <= 0) or abs(np.linalg.det(axes)) < 1e-12:
        raise ValueError("空间变换退化")  # 退化网格不能用于体积或距离计算。
    unit_axes = axes / spacing  # 去除各轴长度，只保留方向。
    if not np.allclose(unit_axes.T @ unit_axes, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError("本检查器不支持含剪切的网格")  # EDT 的 sampling 不能表达轴间剪切。
    return image, spacing, abs(np.linalg.det(axes)) / 1000  # 体素体积由 mm³ 换算为 mL。


def binary_mask(image):
    data = image.get_fdata(dtype=np.float64)  # 保留足够精度，避免把非整数悄悄舍入成 1。
    if not np.isfinite(data).all() or not np.isin(data, [0, 1]).all():
        raise ValueError("标签必须严格为有限的 0/1；不自动阈值化")  # 概率图须在上游按约定处理。
    return data.astype(bool)  # 布尔数组用于集合运算。


def evaluate(paths):
    loaded = [load_geometry(p) for p in paths]  # 顺序为原始 CT、参考标签、预测标签。
    image, spacing, voxel_ml = loaded[0]  # 以原始 CT 的网格作为检查基准。
    for other, _, _ in loaded[1:]:
        if other.shape != image.shape or not np.allclose(other.affine, image.affine, rtol=0, atol=1e-3):
            raise ValueError("形状或 affine 与原始 CT 不一致")  # 停止评分，不在此处重采样。
    ref, pred = [binary_mask(item[0]) for item in loaded[1:]]  # 只加载两个掩膜的完整数组。
    nr, npred = int(ref.sum()), int(pred.sum())  # 统计参考与预测的前景体素数。
    intersection = int(np.count_nonzero(ref & pred))  # 统计同时被判为器官的体素。
    dice = 2 * intersection / (nr + npred) if nr else None  # 仅在参考非空时纳入本例 Dice 汇总。
    if nr:
        status = "both_nonempty" if npred else "prediction_empty"  # 参考存在：区分正常与完全漏检。
    else:
        status = "reference_empty_fp" if npred else "both_empty"  # 参考为空：区分误报与双空。
    hd95 = hdmax = None  # 空掩膜的表面距离保持未定义，不能写成零。
    if nr and npred:
        structure = ndi.generate_binary_structure(3, 1)  # 使用三维六邻接定义边界。
        er = ref & ~ndi.binary_erosion(ref, structure=structure, border_value=0)  # 参考边界体素。
        ep = pred & ~ndi.binary_erosion(pred, structure=structure, border_value=0)  # 预测边界体素。
        dpr = ndi.distance_transform_edt(~er, sampling=spacing)[ep]  # 每个预测边界中心到参考边界的距离。
        drp = ndi.distance_transform_edt(~ep, sampling=spacing)[er]  # 反向计算，补上参考边界的遗漏。
        p95_pred = float(np.percentile(dpr, 95, method="linear"))  # 预测侧边界距离的 95 分位数。
        p95_ref = float(np.percentile(drp, 95, method="linear"))  # 参考侧边界距离的 95 分位数。
        hd95 = max(p95_pred, p95_ref)  # 取两个方向 95 分位数的较大者。
        hdmax = max(dpr.max(), drp.max())  # 同时保留最大距离，避免忽略少量远端孤岛。
    return dict(
        case_status=status,
        dice=dice,
        hd95_mm=hd95,
        hdmax_mm=hdmax,
        reference_ml=nr * voxel_ml,
        prediction_ml=npred * voxel_ml,
        signed_volume_error_ml=(npred - nr) * voxel_ml,  # 正值表示预测体积偏大。
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)  # 提供基本命令行帮助。
    parser.add_argument("manifest", type=Path)  # 输入包含病例信息和三条路径的 CSV。
    parser.add_argument("report", type=Path)  # 输出 JSON，保存成功与失败病例。
    args = parser.parse_args()  # 解析实参。
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)  # 依据首行列名读取每个病例。
        required = {"case_id", "patient_id", "image", "label", "prediction"}
        if not required.issubset(reader.fieldnames or []):
            parser.error("清单缺少必要列")  # 清单结构不完整时整体停止。
        rows = list(reader)  # 固定本次请求集合，不能只保留可评分的病例。
    # 缺失字段在 csv.DictReader 中表现为 None，先确认存在且为字符串，再判断是否空白。
    if any(
        not isinstance(row.get(k), str) or not row[k].strip()
        for row in rows for k in required
    ):
        parser.error("清单存在缺失或空白字段")  # 缺列的行不能默认为空字符串继续评分。
    for row in rows:  # 编号统一去除两端空白后，再进入重复检查、患者计数与报告输出。
        row["case_id"] = row["case_id"].strip()
        row["patient_id"] = row["patient_id"].strip()
    if not rows or len({row["case_id"] for row in rows}) != len(rows):
        parser.error("清单为空或病例编号重复")  # 患者编号可以重复，病例编号必须唯一。
    results = []  # 按清单顺序保留每一个病例的结局。
    for row in rows:
        result = dict(case_id=row["case_id"], patient_id=row["patient_id"])  # 不把身份字段当作医学内容。
        try:
            paths = [(args.manifest.parent / row[k]).resolve() for k in ("image", "label", "prediction")]
            result.update(evaluate(paths))  # 完成几何门控与质量评分。
            digests = dict(zip(("image", "label", "prediction"), map(sha256, paths)))
            result.update(status="ok", sha256=digests)  # 摘要帮助追踪具体文件版本。
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")  # 错误病例保留在总分母中。
        results.append(result)  # 无论成功失败，都写入病例记录。
    ok = [r for r in results if r["status"] == "ok"]  # 区分成功评分数量与请求总数。
    dice = [r["dice"] for r in ok if r["dice"] is not None]  # 参考非空，包含完全漏检的零分。
    hd = [r["hd95_mm"] for r in ok if r["hd95_mm"] is not None]  # 表面距离只在双方非空时有定义。
    summary = dict(
        n_requested=len(rows),  # 本次原计划处理多少病例。
        n_ok=len(ok),  # 多少病例可以按本篇规则评分，不代表分割合格。
        n_error=len(rows) - len(ok),
        n_patients=len({r["patient_id"] for r in rows}),  # 按规范化后的患者编号计数。
        n_dice_defined=len(dice),  # Dice 均值实际使用的病例数。
        dice_mean=float(np.mean(dice)) if dice else None,
        n_hd95_defined=len(hd),  # HD95 均值实际使用的病例数。
        hd95_mean_mm=float(np.mean(hd)) if hd else None,
    )
    for state in ("prediction_empty", "reference_empty_fp", "both_empty"):
        summary["n_" + state] = sum(r["case_status"] == state for r in ok)  # 显式公布空掩膜数量。
    report = dict(
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, nibabel=nib.__version__),
        manifest_sha256=sha256(args.manifest),
        metric_definition="binary foreground; reference-empty Dice=null; 6-neighbour boundary centres; max directional linear P95; mm",
        performance="not measured by this CPU checker", summary=summary, cases=results)
    args.report.parent.mkdir(parents=True, exist_ok=True)  # 创建报告目录。
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))  # 输出紧凑摘要，完整依据仍在 JSON 中。
    return int(summary["n_error"] > 0)  # 非零退出码表示文件检查错误，不是质量是否达标。


if __name__ == "__main__":
    raise SystemExit(main())  # 使命令行能感知失败病例。
