"""26 万个 CUDA 线程，如何推动一滴墨水？（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：NumPy。
"""
import numpy as np


def build_field(N, omega0=0.35, R_ratio=0.35):
    """差异旋转速度场 + 中心右侧的高斯墨滴；固定零边界。"""
    y, x = np.mgrid[0:N, 0:N].astype(np.float64)
    center_x = (N - 1) / 2.0
    center_y = (N - 1) / 2.0
    dx = x - center_x
    dy = y - center_y
    R = N * R_ratio
    omega = omega0 * np.exp(-(dx * dx + dy * dy) / (R * R))
    velocity_x = -omega * dy
    velocity_y = omega * dx

    blob_x = center_x + N * 0.22
    blob_y = center_y
    sigma = N * 0.055
    dye = np.exp(
        -((x - blob_x) ** 2 + (y - blob_y) ** 2) / (2.0 * sigma ** 2)
    )
    dye[0, :] = dye[-1, :] = dye[:, 0] = dye[:, -1] = 0.0
    return velocity_x, velocity_y, dye


def advect(src, velocity_x, velocity_y, dt, N):
    """半拉格朗日平流（反向回溯 + 双线性插值），边界保持 0。"""
    dst = np.zeros_like(src)
    iy, ix = np.indices((N, N))
    sx = np.clip(ix - velocity_x * dt, 0.0, N - 1 - 1e-4)
    sy = np.clip(iy - velocity_y * dt, 0.0, N - 1 - 1e-4)
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    x1 = np.minimum(x0 + 1, N - 1)
    y1 = np.minimum(y0 + 1, N - 1)
    fx = sx - x0
    fy = sy - y0
    top = src[y0, x0] * (1 - fx) + src[y0, x1] * fx
    bottom = src[y1, x0] * (1 - fx) + src[y1, x1] * fx
    dst[1:-1, 1:-1] = (top * (1 - fy) + bottom * fy)[1:-1, 1:-1]
    return dst


def diffuse(src, alpha, N):
    """五点 Laplacian 显式扩散，非负裁剪，边界保持 0。"""
    dst = np.zeros_like(src)
    inner = src[1:-1, 1:-1]
    lap = (
        src[0:-2, 1:-1] + src[2:, 1:-1]
        + src[1:-1, 0:-2] + src[1:-1, 2:]
        - 4 * inner
    )
    dst[1:-1, 1:-1] = np.clip(inner + alpha * lap, 0.0, None)
    return dst


def one_step(dye, velocity_x, velocity_y, dt, alpha, N):
    """一个时间步 = 先平流、再扩散（与 GPU 两个 Kernel 完全对应）。"""
    return diffuse(advect(dye, velocity_x, velocity_y, dt, N), alpha, N)


if __name__ == "__main__":
    N, steps, dt, alpha = 64, 300, 0.02, 0.02
    vx, vy, dye = build_field(N)
    total0 = dye.sum()
    for k in range(steps):
        dye = one_step(dye, vx, vy, dt, alpha, N)
        assert np.isfinite(dye).all(), f"step {k}: 出现 NaN/Inf"
        assert dye.min() >= -1e-12, f"step {k}: 负浓度 {dye.min()}"
    print(f"N={N} 步数={steps}：有限性与非负检查通过")
    print(f"总浓度：初始 {total0:.4f} → 最终 {dye.sum():.4f}（非严格守恒，见正文 8.1）")
