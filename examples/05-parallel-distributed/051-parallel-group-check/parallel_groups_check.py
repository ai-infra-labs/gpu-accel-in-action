"""八张显卡，怎样同时分数据、分层、分矩阵？（第5章 · 并行与分布式）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：仅标准库。
"""
from math import isclose

D, P, T = 2, 2, 2
b, m = 2, 4
world_size = D * P * T

def rank(d, p, t):
    return (d * P + p) * T + t

tp_groups = [[rank(d, p, t) for t in range(T)]
             for d in range(D) for p in range(P)]
pp_groups = [[rank(d, p, t) for p in range(P)]
             for d in range(D) for t in range(T)]
dp_groups = [[rank(d, p, t) for d in range(D)]
             for p in range(P) for t in range(T)]

for groups in (tp_groups, pp_groups, dp_groups):
    assert sorted(r for group in groups for r in group) == list(range(world_size))

a = [1.0, 2.0]
v = [3.0, 4.0]
samples = list(range(1, b * m * D + 1))
local_data = [samples[d * b * m:(d + 1) * b * m] for d in range(D)]
assert len(set(local_data[0]) & set(local_data[1])) == 0

# 每个 rank 只有一个参数的梯度；先用本副本的 8 条样本归一化。
grad = [0.0] * world_size
local_loss = [0.0] * D
for d in range(D):
    for j in range(m):
        microbatch = local_data[d][j * b:(j + 1) * b]
        for x in microbatch:
            # PP 前半段，两张 TP 卡各生成一个隐藏特征。
            h = [a[t] * x for t in range(T)]
            # PP 传 h[t] 给对应的后半段 TP 卡。
            partial = [v[t] * h[t] for t in range(T)]
            prediction = sum(partial)  # TP SUM，不能除以 T。
            error = prediction - 2.0 * x
            # 同一预测即使复制到 TP 卡，也只计一次逻辑 loss。
            local_loss[d] += error ** 2 / (b * m)
            delta = 2.0 * error / (b * m)
            for t in range(T):
                grad[rank(d, 1, t)] += delta * h[t]
                dh = delta * v[t]  # PP 反向传回的激活梯度。
                grad[rank(d, 0, t)] += dh * x

# 普通 DP：同一参数的两个副本梯度平均。
synced = grad.copy()
for group in dp_groups:
    mean_grad = sum(grad[r] for r in group) / D
    for r in group:
        synced[r] = mean_grad

# 未切分的完整网络，直接对全部 16 条样本求解析梯度。
reference = [0.0] * (P * T)
reference_loss = 0.0
for x in samples:
    prediction = x * sum(a[t] * v[t] for t in range(T))
    error = prediction - 2.0 * x
    reference_loss += error ** 2 / len(samples)
    delta = 2.0 * error / len(samples)
    for t in range(T):
        reference[t] += delta * v[t] * x
        reference[T + t] += delta * a[t] * x

for d in range(D):
    for p in range(P):
        for t in range(T):
            assert isclose(synced[rank(d, p, t)], reference[p * T + t],
                           rel_tol=1e-12, abs_tol=1e-12)
assert isclose(sum(local_loss) / D, reference_loss)

# 同步完成后才更新；所有梯度都来自更新前的同一版本参数。
lr = 1e-5
initial = [a[t] if p == 0 else v[t]
           for d in range(D) for p in range(P) for t in range(T)]
updated = [initial[r] - lr * synced[r] for r in range(world_size)]
for group in dp_groups:
    assert all(isclose(updated[r], updated[group[0]]) for r in group)

print("TP groups:", tp_groups)
print("PP groups:", pp_groups)
print("DP groups:", dp_groups)
print("Global batch:", len(samples))
print("Loss:", f"{reference_loss:.6f}")
print("Gradient [a0, a1, v0, v1]:", reference)
print("Updated replica A:", [round(x, 5) for x in updated[:4]])
print("PASS: split gradients match the full-model reference")
