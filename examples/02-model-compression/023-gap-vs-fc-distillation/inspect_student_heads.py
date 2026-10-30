"""参数缩到原来的 1/106，计算量却只少了 27%：小模型到底小在哪？（第2章 · 模型压缩）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyTorch。
"""
import copy
import torch
import torch.nn as nn


class StudentCNN(nn.Module):
    """同一卷积主体，切换 FC 或 GAP 分类头。"""

    def __init__(self, head="fc"):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # [B, 32, 32, 32]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # [B, 32, 16, 16]
            nn.Conv2d(32, 64, 3, padding=1), # [B, 64, 16, 16]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # [B, 64, 8, 8]
        )
        if head == "fc":
            self.classifier = nn.Sequential(
                nn.Flatten(1),             # 将每个样本展开为 4096 维
                nn.Linear(64 * 8 * 8, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 10),
            )
        elif head == "gap":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),   # 每个通道的空间维取平均
                nn.Flatten(1),             # 保留 batch，得到 [B, 64]
                nn.Linear(64, 10),         # 返回 logits，不加 softmax
            )
        else:
            raise ValueError("head 必须为 'fc' 或 'gap'")

    def forward(self, x):
        # 不在这里提前 flatten，空间池化需要完整的四维特征。
        return self.classifier(self.features(x))


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    torch.manual_seed(42)
    original = StudentCNN("fc")
    compact = StudentCNN("gap")

    # 两种结构使用相同卷积初值；不同分类头无法逐参数共享初值。
    compact.features.load_state_dict(original.features.state_dict())

    # 同一结构内复制整个模型，作为 CE/KD 两组的共同起点。
    models = {
        "FC-CE": original,
        "FC-KD": copy.deepcopy(original),
        "GAP-CE": compact,
        "GAP-KD": copy.deepcopy(compact),
    }
    expected = {"FC-CE": 2122186, "FC-KD": 2122186,
                "GAP-CE": 20042, "GAP-KD": 20042}
    x = torch.randn(2, 3, 32, 32)  # 两张随机图片，仅供形状检查

    with torch.inference_mode():
        for name, model in models.items():
            model.eval()
            output = model(x)
            total = count_params(model)
            # 按当前实际 dtype 计算参数字节数，不统计激活和运行时。
            nbytes = sum(p.numel() * p.element_size()
                         for p in model.parameters())
            assert output.shape == (2, 10)
            assert total == expected[name]
            print(f"{name:6s} | total={total:,} | "
                  f"head={count_params(model.classifier):,} | "
                  f"parameter_bytes={nbytes:,} | shape={tuple(output.shape)}")

    # 验证 GAP 确实等价于对每个通道的空间维取平均。
    feature = torch.randn(2, 64, 8, 8)
    actual = nn.AdaptiveAvgPool2d(1)(feature)
    reference = feature.mean(dim=(2, 3), keepdim=True)
    torch.testing.assert_close(actual, reference)

    print("参数量比值：", count_params(original) / count_params(compact))
    print("以上只是结构检查，不代表训练效果或实际加速比。")


if __name__ == "__main__":
    main()
