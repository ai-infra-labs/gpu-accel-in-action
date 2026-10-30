"""CIFAR-10 知识蒸馏：Teacher → Student-CE 基线 → Student-KD → 对照结果表。

依赖：相互兼容的 PyTorch 与 torchvision；不需要 Numba。
首次运行会下载 CIFAR-10。已有教师权重必须使用相同网络、类别顺序和预处理。
对比组（ce 与 kd）沿用同一份初始权重、各自新建同种子 DataLoader，保证公平性。
本文件保存的 checkpoint 包含 model_state_dict；也支持加载原始 state_dict。
"""
import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class TeacherCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 64, 16, 16]
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 128, 8, 8]
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 256, 4, 4]
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class StudentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 32, 16, 16]
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 64, 8, 8]
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def kd_terms(student_logits, teacher_logits, labels, temperature, alpha):
    """返回 (交叉熵, 缩放后的 KL, 组合损失)。

    此处 KL 方向是 KL(P_teacher || P_student)：教师概率是基准。
    两个学生/教师分支都在这里集中处理，便于 fit() 与自检复用。
    """
    t = temperature
    ce = F.cross_entropy(student_logits, labels)
    kd = F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits.detach() / t, dim=1),
        reduction='batchmean',  # 类别维求和，batch 维求平均
    ) * (t * t)  # 乘 T²；不要再次除以 batch_size 或 T²
    return ce, kd, alpha * ce + (1 - alpha) * kd


class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.7):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature 必须是有限正数')
        if not 0 <= alpha <= 1:
            raise ValueError('alpha 必须位于 [0, 1]，表示硬标签 CE 权重')
        self.temperature, self.alpha = temperature, alpha

    def forward(self, student_logits, teacher_logits, labels):
        # Student 保留梯度；Teacher 在外部 no_grad，内部再 detach 防误用。
        _, _, total = kd_terms(student_logits, teacher_logits, labels,
                               self.temperature, self.alpha)
        return total


def make_loaders(args, device):
    # Normalize 的第二个参数是标准差，不是方差。
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    # CIFAR-10 本身就是 32×32，无需 Resize。
    full_train = datasets.CIFAR10(args.data_dir, train=True,
                                  download=True, transform=transform)
    test = datasets.CIFAR10(args.data_dir, train=False,
                            download=True, transform=transform)
    indices = torch.randperm(len(full_train),
                             generator=torch.Generator().manual_seed(args.seed)).tolist()
    # 从训练集划分验证集；不使用测试集选 epoch/权重。
    train = Subset(full_train, indices[5000:])
    val = Subset(full_train, indices[:5000])
    options = dict(batch_size=args.batch_size, num_workers=args.workers,
                   pin_memory=(device.type == 'cuda'))
    return (
        DataLoader(train, shuffle=True,
                   generator=torch.Generator().manual_seed(args.seed), **options),
        DataLoader(val, shuffle=False, **options),
        DataLoader(test, shuffle=False, **options),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def fit(model, train_loader, val_loader, device, epochs, lr, teacher=None,
        temperature=2.0, alpha=0.7, tag=''):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc, best_state = -1.0, None
    if teacher is not None:
        teacher.eval()
        teacher.requires_grad_(False)

    for epoch in range(1, epochs + 1):
        model.train()  # evaluate 会切到 eval，每个 epoch 要切回 train。
        ce_sum = kd_sum = total_sum = 0.0
        count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if teacher is None:
                # 教师或基线：只有真实标签监督。
                loss = F.cross_entropy(logits, labels)
                ce_sum += loss.item() * labels.size(0)
            else:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                ce, kd, loss = kd_terms(logits, teacher_logits, labels,
                                        temperature, alpha)
                ce_sum += ce.item() * labels.size(0)
                kd_sum += kd.item() * labels.size(0)
            loss.backward()
            optimizer.step()
            total_sum += loss.item() * labels.size(0)
            count += labels.size(0)

        accuracy = evaluate(model, val_loader, device)
        message = f'{tag}Epoch {epoch:02d}/{epochs}' \
                  f' | CE={ce_sum / count:.4f}'
        if teacher is not None:
            message += f' | KD={kd_sum / count:.4f}'
        message += f' | Total={total_sum / count:.4f}' \
                   f' | val_acc={accuracy:.2f}%'
        print(message)
        if accuracy > best_acc:
            best_acc = accuracy
            # clone 避免后续参数更新改变已记录的最优权重。
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_acc


def smoke_test(device):
    """损失函数梯度自检：不下载数据，验证纯 KL 确实产生正确的 Student 梯度。"""
    s = torch.randn(4, 10, device=device, requires_grad=True)
    teacher_logits = torch.randn(4, 10, device=device, requires_grad=True)
    labels = torch.zeros(4, dtype=torch.long, device=device)
    t = 2.0
    loss = DistillationLoss(t, alpha=0.0)(s, teacher_logits, labels)
    loss.backward()
    expected = t / s.size(0) * (
        F.softmax(s.detach() / t, dim=1)
        - F.softmax(teacher_logits.detach() / t, dim=1)
    )
    torch.testing.assert_close(s.grad, expected, rtol=1e-4, atol=1e-6)
    assert teacher_logits.grad is None
    assert torch.isfinite(loss) and s.grad.abs().sum() > 0
    # 验证原始两种 CNN 在 CIFAR-10 输入上的输出形状。
    with torch.no_grad():
        x = torch.randn(2, 3, 32, 32, device=device)
        for cls in (TeacherCNN, StudentCNN):
            assert cls().to(device)(x).shape == (2, 10)
    print('损失函数梯度自检通过：KL 梯度与解析式一致、Teacher 无梯度、输出形状正确。')


def network_grad_test(device):
    """网络级梯度自检：仅用 KD（alpha=0），验证梯度真正流进 Student 参数。"""
    torch.manual_seed(0)
    teacher = TeacherCNN().to(device)
    student = StudentCNN().to(device)
    teacher.eval()
    teacher.requires_grad_(False)  # 教师冻结，梯度必须始终为 None
    x = torch.randn(8, 3, 32, 32, device=device)
    labels = torch.zeros(8, dtype=torch.long, device=device)

    before = {k: v.detach().clone() for k, v in student.state_dict().items()}
    with torch.no_grad():
        teacher_logits = teacher(x)
    student_logits = student(x)
    _, kd, _ = kd_terms(student_logits, teacher_logits, labels, 2.0, 0.0)
    kd.backward()

    # 1) 学生参数确实收到非零、有限的梯度
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0
    # 2) 教师参数没有梯度（requires_grad_(False) 下 grad 恒为 None）
    assert all(p.grad is None for p in teacher.parameters())

    # 3) 执行一步更新后，学生参数确实改变
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    optimizer.step()
    after = student.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    print('网络级梯度自检通过：KD 梯度流入 Student 参数并完成一步更新，Teacher 未受影响。')


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 知识蒸馏对照实验')
    parser.add_argument('--mode',
                        choices=['teacher', 'ce', 'kd', 'compare'],
                        default='compare',
                        help='teacher=只训练教师；ce=仅标签基线；'
                             'kd=蒸馏；compare=完整对照（默认）')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--output-dir', default='./checkpoints')
    parser.add_argument('--teacher-checkpoint', default=None)
    parser.add_argument('--teacher-epochs', type=int, default=20)
    parser.add_argument('--student-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if min(args.teacher_epochs, args.student_epochs, args.batch_size) <= 0:
        parser.error('epochs 和 batch-size 必须大于 0')
    if args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('workers 必须非负，lr 必须是有限正数')
    DistillationLoss(args.temperature, args.alpha)  # 提前校验超参数
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('指定了 CUDA，但当前 PyTorch 无可用 CUDA 设备')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    print(f'使用设备：{device}')
    if args.smoke_test:
        smoke_test(device)
        network_grad_test(device)
        return
    if args.teacher_checkpoint and not Path(args.teacher_checkpoint).is_file():
        parser.error('指定的教师权重文件不存在')

    # ---- 数据与共享初始权重 ----
    # 训练 DataLoader 会被各轮训练各自消耗随机状态，因此每组实验都新建独立实例；
    # create 时使用同一个 seed，保证学生 CE / KD 两组的数据顺序一致。
    _, val_loader, test_loader = make_loaders(args, device)
    # 用独立的一份 Student 初值作为两组学生的共同起点。
    init_state = {k: v.detach().cpu().clone()
                  for k, v in StudentCNN().state_dict().items()}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models, results = {}, {}

    # ---- 1. 教师：加载或监督训练 ----
    teacher = None
    if args.mode in ('teacher', 'kd', 'compare'):
        teacher = TeacherCNN().to(device)
        if args.teacher_checkpoint:
            checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu',
                                    weights_only=True)
            teacher.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            print('已加载教师权重；请确认其训练预处理与本脚本一致。')
        else:
            print('开始监督训练 Teacher')
            train_loader, _, _ = make_loaders(args, device)
            fit(teacher, train_loader, val_loader, device,
                args.teacher_epochs, args.lr, tag='[Teacher] ')
            torch.save({'model_state_dict': teacher.state_dict(),
                        'val_accuracy': evaluate(teacher, val_loader, device),
                        'config': vars(args)}, output / 'teacher_best.pth')
        models['Teacher'] = teacher
        results['Teacher'] = evaluate(teacher, val_loader, device)
        print(f'Teacher 验证准确率：{results["Teacher"]:.2f}%')
        if args.mode == 'teacher':
            print(f'权重已保存至：{output.resolve()}')
            return

    # ---- 2. 学生基线：相同初始权重，仅使用标签 ----
    if args.mode in ('ce', 'compare'):
        print('\n开始训练 Student-CE（相同初始权重，仅使用标签）')
        student_ce = StudentCNN().to(device)
        student_ce.load_state_dict(init_state)
        train_loader, _, _ = make_loaders(args, device)
        val_acc = fit(student_ce, train_loader, val_loader, device,
                      args.student_epochs, args.lr, tag='[CE] ')
        torch.save({'model_state_dict': student_ce.state_dict(),
                    'val_accuracy': val_acc, 'config': vars(args)},
                   output / 'student_ce_best.pth')
        models['Student-CE'] = student_ce
        results['Student-CE'] = val_acc

    # ---- 3. 学生蒸馏：相同初始权重，标签 + 教师软目标 ----
    if args.mode in ('kd', 'compare'):
        if teacher is None:
            parser.error('--mode kd 需要先提供教师（--teacher-checkpoint 或先跑 teacher）')
        print('\n开始蒸馏 Student-KD（相同初始权重，标签 + 教师）')
        student_kd = StudentCNN().to(device)
        student_kd.load_state_dict(init_state)
        train_loader, _, _ = make_loaders(args, device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        val_acc = fit(student_kd, train_loader, val_loader, device,
                      args.student_epochs, args.lr, teacher,
                      args.temperature, args.alpha, tag='[KD] ')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        print(f'蒸馏耗时（含每轮验证）：{time.perf_counter() - start:.2f} 秒')
        torch.save({'model_state_dict': student_kd.state_dict(),
                    'val_accuracy': val_acc, 'config': vars(args)},
                   output / 'student_kd_best.pth')
        models['Student-KD'] = student_kd
        results['Student-KD'] = val_acc

    # ---- 4. 统一结果表（验证集 + 独立测试集） ----
    print('\n==== 对照结果（同一初始权重、同一数据顺序）====')
    for name, val_acc in results.items():
        test_acc = evaluate(models[name], test_loader, device)
        print(f'{name:<12} val_acc={val_acc:6.2f}%  test_acc={test_acc:6.2f}%')
    print(f'\n权重已保存至：{output.resolve()}')


if __name__ == '__main__':
    main()
