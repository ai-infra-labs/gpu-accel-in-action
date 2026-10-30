"""NumPy浮点教学检查：不模拟完整GPU或PyTorch AMP执行。"""
import numpy as np

# 小数太小，直接保存到FP16时可能变成0。
grad = np.float32(1e-8)
direct = np.float32(np.float16(grad))
scale = np.float32(1024)
restored = np.float32(np.float16(grad * scale)) / scale
assert direct == 0 and restored > 0
np.testing.assert_allclose(restored, grad, rtol=0.01, atol=0)

# 两个FP32分数不同，转成FP16后可能舍入到同一个数。
logits = np.array([1.0, 1.0001], dtype=np.float32)
rounded = logits.astype(np.float16).astype(np.float32)
assert logits.argmax() == 1 and rounded.argmax() == 0
assert rounded[0] == rounded[1]  # NumPy argmax在并列时选首个位置。

# 刻意用FP16进行归约会溢出；这不是说AMP一定按此方式归约。
with np.errstate(over="ignore"):
    half_sum = np.ones(96 ** 3, dtype=np.float16).sum(dtype=np.float16)
full_sum = np.ones(96 ** 3, dtype=np.float16).sum(dtype=np.float32)
assert np.isinf(half_sum) and full_sum == 96 ** 3
print("直接/缩放后恢复的数", direct, restored)
print("舍入前/后的分数", logits, rounded)
print("FP16/FP32归约", half_sum, full_sum)
print("PASS：仅验证格式舍入、范围与教学缩放，不是GPU实验。")
