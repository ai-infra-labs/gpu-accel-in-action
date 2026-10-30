#!/usr/bin/env python3
"""由 docs/_data/articles.json 生成 docs/reference/article-index.md。

设计：保留已提交的「表头 / 状态含义图例 / 统计行 / 提取已完成」等策展段落
（按原文件逐字保留，修改前言请直接改 docs/reference/article-index.md），**只重生成
那张 81 行的映射数据表**。这样既不会丢掉人工写的前言，又消除了手工维护大表
带来的漂移（这正是之前 待验证/待环境 残留的根源）。

用法：
    python3 scripts/gen_index.py            # 写 docs/reference/article-index.md
    python3 scripts/gen_index.py --check    # 只比对，过期则退出码 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import REPO_ROOT, ARTICLE_INDEX  # noqa: E402

MANIFEST = REPO_ROOT / "docs" / "_data" / "articles.json"
TABLE_HEADER = "| 篇号 | 标题 | 代码块 | 最大块 | 状态 | 目录 |"
TABLE_SEP = "|---|---|---:|---:|---|"

# 找不到已提交文件时的兜底前言（首跑用，之后以仓库文件为准）
FALLBACK_PREAMBLE = (
    "# 81 篇 → 仓库目录 映射与进度\n"
    "\n"
    "> 按章节阅读请看 [`CURRICULUM.md`](CURRICULUM.md)；每篇的依赖与实测状态见 [`ATLAS.md`](ATLAS.md)。\n"
    "\n"
    "本表是提取作业清单。状态含义：\n"
    "\n"
    "| 状态 | 含义 | 处理方式 |\n"
    "|---|---|---|\n"
    "| 已入库（已实测） | 代码已入库且跑通，数字记在 expected_output.md | 无需处理 |\n"
    "| 已入库（已实测·合成数据） | 脚本跑通，但用的是合成输入而非真实病例 | 有真实数据时补一份 |\n"
    "| 已入库（部分实测） | 多文件篇目，一部分已跑通、另一部分未实测 | 只补未验证的那部分 |\n"
    "| 已入库（无独立代码） | 只留 README 说明 | 无需处理 |\n"
    "| 待提取 | 有 ≥60 行完整代码块，可机械拆分 | `scripts/extract_block.py` + pyflakes |\n"
    "| 待拼装 | 最大代码块 20–59 行，或长块缺 import | 人工合并 v0→v1 |\n"
    "| 待评估 | 最大代码块 <20 行，疑似无独立代码 | 逐篇判断 |\n"
)


def load_articles() -> list[dict]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data["articles"]


def preamble() -> tuple[str, bool]:
    """取已提交前言（含表格头与分隔行）；返回 (前言文本, 是否找到分隔行)。

    找不到分隔行（首次运行无提交文件）时，用兜底前言 + 重建表头。
    """
    if not ARTICLE_INDEX.is_file():
        return FALLBACK_PREAMBLE.rstrip("\n") + "\n" + TABLE_HEADER + "\n" + TABLE_SEP, False
    lines = ARTICLE_INDEX.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(lines):
        if line.strip() == TABLE_SEP:
            # 保留数据表分隔行（含）之前的所有内容，逐字复刻，避免丢失空行
            return "\n".join(lines[: i + 1]), True
    return FALLBACK_PREAMBLE.rstrip("\n") + "\n" + TABLE_HEADER + "\n" + TABLE_SEP, False


def render() -> str:
    arts = sorted(load_articles(), key=lambda a: a["id"])
    rows = [
        f"| {a['id']:03d} | {a['title']} | {a['block_count']} | {a['max_block']} "
        f"| {a['ingest_status']} | {a['repo_dir'] or '—'} |"
        for a in arts
    ]
    pre, _ = preamble()
    return pre + "\n" + "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只比对，不写文件")
    args = parser.parse_args()

    text = render()
    if args.check:
        if not ARTICLE_INDEX.is_file():
            print(f"缺少 {ARTICLE_INDEX.relative_to(REPO_ROOT)}，先运行 gen_index.py")
            return 1
        if ARTICLE_INDEX.read_text(encoding="utf-8") != text:
            print("docs/reference/article-index.md 已过期，请重新生成：python3 scripts/gen_index.py")
            return 1
        print("docs/reference/article-index.md 与仓库一致。")
        return 0

    ARTICLE_INDEX.write_text(text, encoding="utf-8")
    print(f"已写入 {ARTICLE_INDEX.relative_to(REPO_ROOT)}（{len(load_articles())} 篇）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
