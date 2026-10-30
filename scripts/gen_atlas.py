#!/usr/bin/env python3
"""生成 docs/reference/capability-matrix.md：53 个篇目的能力矩阵。

数据全部来自仓库自身，不手工维护：

  篇号/目录  —— 目录名
  代码       —— 目录内代码文件数与总行数
  依赖层/硬件 —— 复用 static_check.py 的分层与硬件判定
  验证状态   —— expected_output.md 第一段的「**状态：…**」
  前置       —— README.md 的「## 前置」小节

所以这张表不会和代码漂移：改了代码或文档，重跑一次就对齐。

用法：
    python3 scripts/gen_atlas.py            # 写 docs/reference/capability-matrix.md
    python3 scripts/gen_atlas.py --check    # 只比对，过期则退出码 1
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ATLAS = REPO_ROOT / "docs" / "reference" / "capability-matrix.md"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import static_check  # noqa: E402  （同目录的工具脚本，复用它的分层规则）

CODE_SUFFIXES = {".py", ".cu", ".c", ".cuh", ".cpp"}

# 章节 / 机制标签 / 标签含义：统一收口到 scripts/corpus.py，避免词表漂移。
from corpus import CHAPTERS, MECHANISMS, LEGEND  # noqa: E402

TIER_LABEL = {"advanced": "特殊", "gpu": "GPU", "cpu": "CPU", "stdlib": "无"}


def chapter_of(num: int) -> tuple[int, str]:
    for idx, name, _slug, lo, hi in CHAPTERS:
        if lo <= num <= hi:
            return idx, name
    return 0, "未分章"


def read_section(path: Path, title: str) -> str:
    """取 README 某个二级小节的第一段纯文本，用于 ATLAS 的「前置」列。"""
    if not path.is_file():
        return "—"
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(title)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m:
        return "—"
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if line and not line.startswith(("|", "-", "`", "#")):
            return line
    return "—"


def read_status(path: Path) -> str:
    if not path.is_file():
        return "无 expected_output"
    text = path.read_text(encoding="utf-8")
    m = re.search(r"\*\*状态：(.+?)\*\*", text)
    if not m:
        return "未声明"
    status = m.group(1).strip().rstrip("。").replace("`", "")
    return status if len(status) <= 34 else status[:33] + "…"


def scan_dir(path: Path) -> tuple[int, int, str, str]:
    """返回（文件数、行数、依赖层、硬件需求）。"""
    files = [f for f in path.iterdir() if f.suffix in CODE_SUFFIXES]
    lines = sum(len(f.read_text(encoding="utf-8").splitlines()) for f in files)
    tiers, needs_gpu = [], False
    for f in files:
        if f.suffix not in {".py"}:
            continue
        source = f.read_text(encoding="utf-8")
        names = static_check.top_level_imports(ast.parse(source, filename=str(f)))
        tier, _ = static_check.classify(names)
        tiers.append(tier)
        needs_gpu = needs_gpu or static_check.needs_gpu(source)
    if not tiers:  # 只有 .cu/.cpp 的篇目按 GPU 层算
        return len(files), lines, "gpu", "需 GPU" if files else "不适用"
    order = {t: i for i, t in enumerate(static_check.TIER_ORDER)}
    tier = sorted(tiers, key=lambda t: order.get(t, 99))[0]
    return len(files), lines, tier, "需 GPU" if needs_gpu else "CPU 可跑"


def render() -> str:
    rows = []
    articles = sorted(
        p for p in (REPO_ROOT / "examples").glob("[0-9][0-9]-*/*")
        if p.is_dir() and re.match(r"^\d{3}-", p.name)
    )
    for path in articles:
        num = int(path.name[:3])
        ch, ch_name = chapter_of(num)
        n_files, n_lines, tier, hw = scan_dir(path)
        label = TIER_LABEL.get(tier, tier)
        if tier == "cpu" and hw.startswith("需"):
            label = "CPU*"  # 装 CPU 层的依赖不够，还得有卡
        rows.append((num, ch, ch_name, path.name, n_files, n_lines, label, hw,
                     read_status(path / "expected_output.md"),
                     read_section(path / "README.md", "前置")))

    out = [
        "# 能力矩阵（自动生成，勿手改）",
        "",
        "由 `python3 scripts/gen_atlas.py` 从目录、`README.md`、`expected_output.md` 与",
        "`scripts/static_check.py` 汇总而成。改完代码或文档重跑一次即可；",
        "`python3 scripts/gen_atlas.py --check` 会在表过期时报错。",
        "",
        f"共 {len(rows)} 个篇目。读表三条约定：",
        "",
        "- **依赖层** CPU / GPU / 特殊 对应 `requirements/` 的三个文件；`CPU*` 表示脚本依赖在",
        "  CPU 层，但调用了 CUDA API——装完依赖仍需显卡。",
        "- **硬件** 写“CPU 可跑”不代表已实测，只代表不强制要卡。",
        "- **验证状态** 取自各篇 `expected_output.md` 的首行声明，是唯一算数的列。",
        "",
        "| 章 | 篇 | 目录 | 代码 | 依赖层 | 硬件 | 验证状态 | 机制 | 前置 |",
        "|---:|---:|---|---|---|---|---|---|---|",
    ]
    for num, ch, _, name, n_files, n_lines, tier, hw, status, pre in rows:
        code = f"{n_files} 个 / {n_lines} 行" if n_files else "无代码"
        tags = "、".join(MECHANISMS.get(num, [])) or "—"
        out.append(f"| {ch} | {num:03d} | `{name}` | {code} | {tier} | {hw} | {status} | {tags} | {pre} |")

    out += ["", "## 各章速览", ""]
    for idx, name, slug, lo, hi in CHAPTERS:
        in_ch = [r for r in rows if r[1] == idx]
        if not in_ch:
            continue
        tested = sum(1 for r in in_ch if "已实测" in r[8])
        out.append(f"- **第 {idx} 章 {name}**（`{slug}/`，{lo:03d}–{hi:03d}，"
                   f"收录 {len(in_ch)} 篇，已实测 {tested} 篇）")

    # 按机制反查：同一机制常常横跨多个章节，这是章目录给不了的视角
    out += ["", "## 按机制反查", "",
            "标签含义见下表；同一篇可带 2–3 个标签，所以各项之和大于篇数。", ""]
    index: dict[str, list[int]] = {}
    for num, tags in MECHANISMS.items():
        for tag in tags:
            index.setdefault(tag, []).append(num)
    for tag, nums in sorted(index.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        arts = "、".join(f"{n:03d}" for n in sorted(nums))
        out.append(f"- **`{tag}`**（{LEGEND.get(tag, '未定义')}，{len(nums)} 篇）：{arts}")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只比对，不写文件")
    args = parser.parse_args()

    text = render()
    if args.check:
        if not ATLAS.is_file():
            print(f"缺少 {ATLAS.relative_to(REPO_ROOT)}，先运行 gen_atlas.py")
            return 1
        if ATLAS.read_text(encoding="utf-8") != text:
            print("docs/reference/capability-matrix.md 已过期，请重新生成：python3 scripts/gen_atlas.py")
            return 1
        print("docs/reference/capability-matrix.md 与仓库一致。")
        return 0

    ATLAS.write_text(text, encoding="utf-8")
    print(f"已写入 {ATLAS.relative_to(REPO_ROOT)}（{text.count(chr(10)) - 1} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
