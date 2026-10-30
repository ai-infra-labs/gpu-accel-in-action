"""1024 个 CUDA 线程一起模拟引力，为什么算出的'宇宙'可能是错的？（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import time

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


KERNEL_CODE = r"""
__global__ void compute_acceleration(
    const float *positions,
    float *accelerations,
    int n,
    float softening2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int base = 3 * idx;
    float px = positions[base];
    float py = positions[base + 1];
    float pz = positions[base + 2];

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    for (int j = 0; j < n; ++j) {
        int source = 3 * j;
        float dx = positions[source]     - px;
        float dy = positions[source + 1] - py;
        float dz = positions[source + 2] - pz;

        float r2 = dx * dx + dy * dy + dz * dz + softening2;
        float inv_r = rsqrtf(r2);
        float inv_r3 = inv_r * inv_r * inv_r;

        ax += dx * inv_r3;
        ay += dy * inv_r3;
        az += dz * inv_r3;
    }

    accelerations[base]     = ax;
    accelerations[base + 1] = ay;
    accelerations[base + 2] = az;
}


__global__ void integrate(
    float *positions,
    float *velocities,
    const float *accelerations,
    int n,
    float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int base = 3 * idx;

    float vx = velocities[base]     + accelerations[base]     * dt;
    float vy = velocities[base + 1] + accelerations[base + 1] * dt;
    float vz = velocities[base + 2] + accelerations[base + 2] * dt;

    velocities[base]     = vx;
    velocities[base + 1] = vy;
    velocities[base + 2] = vz;

    positions[base]     += vx * dt;
    positions[base + 1] += vy * dt;
    positions[base + 2] += vz * dt;
}
"""


module = SourceModule(KERNEL_CODE, options=["-O3"])
compute_acceleration = module.get_function("compute_acceleration")
integrate = module.get_function("integrate")


def cpu_step(positions, velocities, dt, softening2):
    """小规模 CPU Reference：执行一个半隐式 Euler 时间步。

    只用于验证数学结果，不作为经过优化的 CPU 性能基线；
    它外层是 Python 循环 + NumPy，不适合与 GPU 比较毫秒数。
    """
    positions = positions.copy()
    velocities = velocities.copy()
    accelerations = np.zeros_like(positions)

    for i in range(positions.shape[0]):
        delta = positions - positions[i]
        r2 = np.einsum("ij,ij->i", delta, delta) + softening2
        inv_r3 = 1.0 / np.power(r2, 1.5)

        # 自作用的 delta 为 0，贡献本来就是 0；显式置零更易读。
        inv_r3[i] = 0.0
        accelerations[i] = np.sum(delta * inv_r3[:, None], axis=0)

    velocities += accelerations * dt
    positions += velocities * dt
    return positions.astype(np.float32), velocities.astype(np.float32)


def gpu_simulate(initial_positions, initial_velocities, steps, dt, softening2,
                 warmup=True):
    """数据常驻 GPU，连续执行多个时间步。"""
    positions = np.ascontiguousarray(initial_positions, dtype=np.float32)
    velocities = np.ascontiguousarray(initial_velocities, dtype=np.float32)

    n = positions.shape[0]
    accelerations = np.empty_like(positions)

    positions_gpu = cuda.mem_alloc(positions.nbytes)
    velocities_gpu = cuda.mem_alloc(velocities.nbytes)
    accelerations_gpu = cuda.mem_alloc(accelerations.nbytes)

    cuda.memcpy_htod(positions_gpu, positions)
    cuda.memcpy_htod(velocities_gpu, velocities)

    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    launch = {
        "block": (block_size, 1, 1),
        "grid": (grid_size, 1, 1),
    }

    def one_step():
        compute_acceleration(
            positions_gpu,
            accelerations_gpu,
            np.int32(n),
            np.float32(softening2),
            **launch,
        )
        integrate(
            positions_gpu,
            velocities_gpu,
            accelerations_gpu,
            np.int32(n),
            np.float32(dt),
            **launch,
        )

    if warmup:
        one_step()
        cuda.Context.synchronize()

        # 预热会改变状态，因此恢复完全相同的初始数据。
        cuda.memcpy_htod(positions_gpu, positions)
        cuda.memcpy_htod(velocities_gpu, velocities)

    start = cuda.Event()
    end = cuda.Event()

    start.record()
    for _ in range(steps):
        one_step()
    end.record()
    end.synchronize()

    kernel_ms = start.time_till(end)

    result_positions = np.empty_like(positions)
    result_velocities = np.empty_like(velocities)
    cuda.memcpy_dtoh(result_positions, positions_gpu)
    cuda.memcpy_dtoh(result_velocities, velocities_gpu)

    return result_positions, result_velocities, kernel_ms


def verify_one_step():
    """使用较小数据集比较 CPU 与 GPU 的一步结果。

    注意：下面的 rtol/atol 只作为当前一步冒烟测试的阈值（固定 N、固定
    dt 与软化项），不是放之四海皆准的容差标准。
    """
    rng = np.random.default_rng(7)
    n = 128
    dt = np.float32(1e-3)
    softening2 = np.float32(1e-4)

    positions = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    velocities = np.zeros((n, 3), dtype=np.float32)

    cpu_positions, cpu_velocities = cpu_step(
        positions, velocities, dt, softening2
    )
    gpu_positions, gpu_velocities, _ = gpu_simulate(
        positions, velocities, steps=1, dt=dt,
        softening2=softening2, warmup=False
    )

    position_error = np.max(np.abs(cpu_positions - gpu_positions))
    velocity_error = np.max(np.abs(cpu_velocities - gpu_velocities))

    print(f"一步位置最大绝对误差: {position_error:.6e}")
    print(f"一步速度最大绝对误差: {velocity_error:.6e}")

    np.testing.assert_allclose(
        gpu_positions, cpu_positions, rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        gpu_velocities, cpu_velocities, rtol=2e-4, atol=2e-5
    )
    print("CPU Reference 与 GPU 结果在容差内一致。")


def main():
    verify_one_step()

    rng = np.random.default_rng(42)
    n = 1024
    steps = 100
    dt = np.float32(1e-3)
    softening2 = np.float32(1e-4)

    positions = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    velocities = np.zeros((n, 3), dtype=np.float32)

    wall_start = time.perf_counter()
    result_positions, result_velocities, kernel_ms = gpu_simulate(
        positions,
        velocities,
        steps=steps,
        dt=dt,
        softening2=softening2,
        warmup=True,
    )
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    print(f"{steps} 步 Kernel 总时间: {kernel_ms:.3f} ms")
    print(f"函数端到端时间: {wall_ms:.3f} ms")
    print("前 5 个粒子的位置：")
    print(result_positions[:5])
    print("前 5 个粒子的速度：")
    print(result_velocities[:5])


if __name__ == "__main__":
    main()
