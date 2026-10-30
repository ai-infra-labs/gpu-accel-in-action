"""26 万个 CUDA 线程，如何推动一滴墨水？（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy、Matplotlib。
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


# ================================================================
# 1. CUDA 内核：平流与扩散（两个 Kernel，对应第 7 章的一阶算子分裂）
# ================================================================

kernel_code = r"""
__global__ void advect(
    const float *dye_old,       // 当前时间步浓度场：只读
    float *dye_advected,        // 平流结果：只写
    const float *velocity_x,    // x 方向速度场：只读
    const float *velocity_y,    // y 方向速度场：只读
    int N,
    float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) {      // 向上取整启动 Grid 的多余线程
        return;
    }
    int i = y * N + x;

    // 固定零边界：避免访问网格外部，也满足下一轮邻域读取的一致性
    if (x == 0 || x == N - 1 || y == 0 || y == N - 1) {
        dye_advected[i] = 0.0f;
        return;
    }

    // 半拉格朗日：沿速度反方向回溯一个时间步（无量纲网格坐标，无
    // 需再除以网格间距 h）
    float sx = (float)x - dt * velocity_x[i];
    float sy = (float)y - dt * velocity_y[i];

    // 钳制到可插值区域内；上界留 1e-4，保证 x1/y1 不越界
    sx = fminf(fmaxf(sx, 0.0f), (float)(N - 1) - 1e-4f);
    sy = fminf(fmaxf(sy, 0.0f), (float)(N - 1) - 1e-4f);

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = min(x0 + 1, N - 1);
    int y1 = min(y0 + 1, N - 1);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    // 包围回溯点的四个旧网格点
    float q00 = dye_old[y0 * N + x0];
    float q10 = dye_old[y0 * N + x1];
    float q01 = dye_old[y1 * N + x0];
    float q11 = dye_old[y1 * N + x1];

    // 双线性插值：先沿 x，再沿 y
    float top    = q00 * (1.0f - fx) + q10 * fx;
    float bottom = q01 * (1.0f - fx) + q11 * fx;
    dye_advected[i] = top * (1.0f - fy) + bottom * fy;
}


__global__ void diffuse(
    const float *dye_advected,  // 平流结果：只读
    float *dye_new,             // 本轮最终浓度：只写
    int N,
    float alpha)                // alpha = D*dt/h^2，无量纲
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) {
        return;
    }
    int i = y * N + x;

    if (x == 0 || x == N - 1 || y == 0 || y == N - 1) {
        dye_new[i] = 0.0f;
        return;
    }

    // 五点 Laplacian，h = 1
    float center = dye_advected[i];
    float left   = dye_advected[y * N + (x - 1)];
    float right  = dye_advected[y * N + (x + 1)];
    float up     = dye_advected[(y - 1) * N + x];
    float down   = dye_advected[(y + 1) * N + x];
    float laplacian = left + right + up + down - 4.0f * center;

    // 非负裁剪：教学演示友好，但会破坏严格守恒（见正文第 8.1 节说明）
    dye_new[i] = fmaxf(center + alpha * laplacian, 0.0f);
}
"""


module = SourceModule(kernel_code, options=["-O3"])
advect = module.get_function("advect")
diffuse = module.get_function("diffuse")


# ================================================================
# 2. 差异旋转速度场与初始染料（第 6.1 节）
# ================================================================

def create_initial_fields(N, omega0=0.35, R_ratio=0.35):
    """差异旋转：omega(r)=omega0*exp(-r^2/R^2)，内外圈转速不同。"""

    y, x = np.mgrid[0:N, 0:N].astype(np.float32)
    center_x = np.float32((N - 1) / 2.0)
    center_y = np.float32((N - 1) / 2.0)

    dx = x - center_x
    dy = y - center_y
    R = np.float32(N * R_ratio)
    omega = omega0 * np.exp(-(dx * dx + dy * dy) / (R * R))

    velocity_x = (-omega * dy).astype(np.float32)
    velocity_y = ( omega * dx).astype(np.float32)

    # 在中心右侧放一团高斯墨滴，便于观察它绕中心旋转、被拉伸成弧
    blob_x = center_x + np.float32(N * 0.22)
    blob_y = center_y
    sigma = np.float32(N * 0.055)

    dye = np.exp(
        -((x - blob_x) ** 2 + (y - blob_y) ** 2) / (2.0 * sigma ** 2)
    ).astype(np.float32)

    # 固定零边界
    dye[0, :] = 0.0
    dye[-1, :] = 0.0
    dye[:, 0] = 0.0
    dye[:, -1] = 0.0

    return velocity_x, velocity_y, dye


# ================================================================
# 3. 主流程封装：run_gpu 可传任意步数（steps=1 用于与 CPU 对拍）
# ================================================================

def run_gpu(velocity_x, velocity_y, dye_initial, dt, alpha, steps,
            block=(16, 16, 1)):
    """在 GPU 上推进 steps 步平流+扩散，返回
    (dye_final, kernel_ms, wall_ms, d2h_ms)。"""
    # 输入统一为 float32 且 C 连续（内核按 const float* 解释显存），
    # 避免调用方传入 float64 时被逐字节误读
    dye_initial = np.ascontiguousarray(dye_initial, dtype=np.float32)
    velocity_x = np.ascontiguousarray(velocity_x, dtype=np.float32)
    velocity_y = np.ascontiguousarray(velocity_y, dtype=np.float32)

    assert dye_initial.ndim == 2
    assert dye_initial.shape[0] == dye_initial.shape[1]
    assert velocity_x.shape == dye_initial.shape
    assert velocity_y.shape == dye_initial.shape
    assert 0.0 <= float(alpha) <= 0.25, "显式扩散稳定性：alpha 必须落在 [0, 0.25]"
    assert steps >= 0

    N = dye_initial.shape[0]

    # 双缓冲：dye_a 保存本轮状态，dye_b 作为平流中转，两步间角色轮换
    dye_a = cuda.to_device(dye_initial)
    dye_b = cuda.mem_alloc(dye_initial.nbytes)

    # 速度场全程不变，只上传一次
    velocity_x_gpu = cuda.to_device(velocity_x)
    velocity_y_gpu = cuda.to_device(velocity_y)

    grid = (
        (N + block[0] - 1) // block[0],
        (N + block[1] - 1) // block[1],
        1,
    )

    # CUDA Event 只测 Kernel 累计时间；墙钟测端到端（含最后一步 D2H）
    start_event = cuda.Event()
    end_event = cuda.Event()
    wall_start = time.perf_counter()
    start_event.record()

    for _ in range(steps):
        # 平流：dye_a(旧) -> dye_b(平流结果)
        advect(
            dye_a, dye_b, velocity_x_gpu, velocity_y_gpu,
            np.int32(N), dt, block=block, grid=grid,
        )
        # 扩散：dye_b(平流结果) -> dye_a(本轮最终值)
        diffuse(
            dye_b, dye_a, np.int32(N), alpha, block=block, grid=grid,
        )

    end_event.record()
    end_event.synchronize()
    kernel_ms = start_event.time_till(end_event)

    # 最终结果位于 dye_a；D2H 计入端到端墙钟
    dye_final = np.empty_like(dye_initial)
    t0_d2h = time.perf_counter()
    cuda.memcpy_dtoh(dye_final, dye_a)
    d2h_ms = (time.perf_counter() - t0_d2h) * 1000.0
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    return dye_final, kernel_ms, wall_ms, d2h_ms


# ================================================================
# 4. 演示运行：512 × 512、300 步（脚本被直接执行时触发）
# ================================================================

if __name__ == "__main__":
    from pathlib import Path

    N, steps = 512, 300
    dt = np.float32(0.02)
    alpha = np.float32(0.02)   # alpha = D*dt/h^2，无量纲，<= 0.25 才稳定
    velocity_x, velocity_y, dye_initial = create_initial_fields(N)

    dye_final, kernel_ms, wall_ms, d2h_ms = run_gpu(
        velocity_x, velocity_y, dye_initial, dt, alpha, steps,
    )

    # 基本数值检查
    assert np.isfinite(dye_final).all(), "结果中出现 NaN 或 Inf"
    assert dye_final.min() >= -1e-6, "浓度出现明显负值"

    print(f"网格：{N} × {N}")
    print(f"时间步：{steps}")
    print(f"累计 GPU 内核时间：{kernel_ms:.3f} ms")
    print(f"平均每步内核时间：{kernel_ms / steps:.6f} ms")
    print(f"最终 D2H 下载时间：{d2h_ms:.3f} ms")
    print(f"仿真循环端到端时间（含最终 D2H；不含上下文初始化/编译/显存分配/初始上传）：{wall_ms:.3f} ms")
    print(f"最终浓度范围：[{dye_final.min():.6f}, {dye_final.max():.6f}]")
    print(f"总浓度比（最终/初始）：{dye_final.sum() / dye_initial.sum():.4f}")
    print("（总浓度略降来自半拉格朗日插值不严格守恒 + 固定零边界 + 浮点舍入；fmaxf 裁剪在本参数下不触发，见正文 8.1）")

    # 可视化初始与最终浓度
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    images = [dye_initial, dye_final]
    titles = ["初始染料浓度", f"经过 {steps} 个时间步"]

    for ax, image, title in zip(axes, images, titles):
        im = ax.imshow(
            image,
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=float(dye_initial.max()),
        )
        ax.set_title(title)
        ax.set_xlabel("x / 网格列")
        ax.set_ylabel("y / 网格行")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    Path("img").mkdir(parents=True, exist_ok=True)
    plt.savefig("img/figure-dye-gpu-512.png", dpi=180)
    plt.show()
