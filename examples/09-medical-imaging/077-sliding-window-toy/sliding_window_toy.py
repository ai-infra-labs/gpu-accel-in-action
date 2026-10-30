"""NumPy教学版三维滑窗：只支持单幅、单输入通道、同分辨率两通道输出。"""
from itertools import product
from math import prod
import numpy as np


def starts(length, roi, overlap):
    # 调用方先补齐短轴，因此这里要求length >= roi。
    if not (isinstance(length, int) and isinstance(roi, int) and length >= roi > 0):
        raise ValueError("轴长度必须不小于正整数窗口尺寸。")
    if not 0 <= overlap < 1:
        raise ValueError("overlap必须位于[0,1)。")
    step = max(int(roi * (1 - overlap)), 1)
    positions = list(range(0, length - roi + 1, step))
    if positions[-1] != length - roi:
        positions.append(length - roi)  # 尾窗贴住末端，不丢掉剩余体素。
    return positions


def weights(roi, mode):
    if mode == "constant":
        return np.ones(roi, dtype=np.float64)
    if mode != "gaussian":
        raise ValueError("只支持constant或gaussian。")
    w = np.ones(roi, dtype=np.float64)
    for axis, size in enumerate(roi):
        coord = np.arange(size) - (size - 1) / 2
        g = np.exp(-0.5 * (coord / (0.125 * size)) ** 2)
        shape = [1, 1, 1]
        shape[axis] = size
        w *= g.reshape(shape)  # 三个轴的一维权重相乘。
    return np.maximum(w, 1e-3)  # 防止角落权重过小；不是模型置信度。


def merge(volume, predictor, roi=(4, 4, 4), overlap=0.5, mode="constant"):
    x = np.asarray(volume)
    # 必须在float64转换前拒绝复数，否则虚部会被丢弃后静默通过。
    if np.iscomplexobj(x):
        raise ValueError("本例只接受实数体积。")
    x = x.astype(np.float64, copy=False)
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("输入必须是非空、有限的三维数组。")
    if len(roi) != 3 or any(not isinstance(r, int) or r < 1 for r in roi):
        raise ValueError("roi必须包含三个正整数。")
    # 小于窗口的轴补0；记录左侧补齐量，最后裁回输入网格。
    missing = [max(r - s, 0) for s, r in zip(x.shape, roi)]
    pads = [(n // 2, n - n // 2) for n in missing]
    padded = np.pad(x, pads, mode="constant", constant_values=0)
    axes = [starts(int(s), r, overlap) for s, r in zip(padded.shape, roi)]
    score_sum = np.zeros((2,) + padded.shape, dtype=np.float64)
    weight_sum = np.zeros(padded.shape, dtype=np.float64)
    w = weights(roi, mode)
    for pos in product(*axes):
        region = tuple(slice(p, p + r) for p, r in zip(pos, roi))
        pred = np.asarray(predictor(padded[region]))
        if np.iscomplexobj(pred):  # 预测器输出沿用同一套实数契约。
            raise ValueError("predictor必须返回实数分数。")
        pred = pred.astype(np.float64, copy=False)
        if pred.shape != (2,) + tuple(roi) or not np.isfinite(pred).all():
            raise ValueError("predictor必须返回有限、同分辨率的两通道分数。")
        score_sum[(slice(None),) + region] += pred * w  # 按通道累加分数。
        weight_sum[region] += w  # 单独累加分母，不能只累计分数。
    assert np.all(weight_sum > 0), "存在漏覆盖位置。"
    merged = score_sum / weight_sum[None]
    crop = tuple(slice(left, left + size) for (left, _), size in zip(pads, x.shape))
    # 只计数，不再重新生成所有起点组合，避免大窗口数时的额外主机内存。
    window_count = prod(len(axis) for axis in axes)
    return merged[(slice(None),) + crop], weight_sum[crop], window_count


def pointwise(patch):
    # 可精确核对的假预测器；没有训练、没有识别器官。
    return np.stack([-patch, patch], axis=0)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    checked = 0
    for shape in [(7, 9, 5), (2, 3, 1), (4, 4, 4), (8, 8, 8)]:
        x = rng.normal(size=shape)
        for overlap in (0, 0.25, 0.5, 0.75):
            for mode in ("constant", "gaussian"):
                out, denominator, _ = merge(x, pointwise, overlap=overlap, mode=mode)
                np.testing.assert_allclose(out, pointwise(x), rtol=1e-12, atol=1e-12)
                assert out.shape == (2,) + shape and np.all(denominator > 0)
                checked += 1
    assert starts(11, 4, 0.5) == [0, 2, 4, 6, 7]

    # 复数体积必须在float64转换前被拒绝。
    for bad in (np.zeros((4, 4, 4), dtype=np.complex128),
                np.array([[[1 + 2j]]])):
        try:
            merge(bad, pointwise)
        except ValueError:
            continue
        raise AssertionError("复数体积没有被拒绝。")

    # 手算融合：同一位置收到分数2和10，权重3和1 → (6+10)/4=4；等权 → 6。
    scores, ws = np.array([2.0, 10.0]), np.array([3.0, 1.0])
    assert abs((scores * ws).sum() / ws.sum() - 4.0) < 1e-12
    assert abs(scores.mean() - 6.0) < 1e-12

    # 让两个重叠窗口给出不同分数，核对实际merge路径：
    # 位置0-1只被首窗覆盖→2，4-5只被尾窗覆盖→10，2-3等权→6。
    calls = {"n": 0}
    def two_scores(patch):
        value = 2.0 if calls["n"] == 0 else 10.0
        calls["n"] += 1
        return np.full((2,) + patch.shape, value)
    merged, _, count = merge(np.zeros((6, 1, 1)), two_scores,
                             roi=(4, 1, 1), overlap=0.5)
    np.testing.assert_allclose(merged[0, :, 0, 0], [2, 2, 6, 6, 10, 10])
    assert count == 2

    # 独立核对gaussian权重本身，避免只用逐点预测器自证。
    t = np.arange(4)
    expected_w = np.maximum(np.exp(-0.5 * ((t - 1.5) / 0.5) ** 2), 1e-3)
    np.testing.assert_allclose(weights((4, 1, 1), "gaussian").ravel(),
                               expected_w, rtol=1e-12, atol=1e-12)

    print(f"PASS: {checked}种合成组合；覆盖、加权归一化、短轴补齐与裁回；"
          "复数拒绝；窗口分歧融合与高斯权重手算。")
    for overlap in (0, 0.25, 0.5, 0.75):
        per_axis = len(starts(256, 96, overlap))
        print("overlap", overlap, "每轴窗口", per_axis, "三维总数", per_axis ** 3)
