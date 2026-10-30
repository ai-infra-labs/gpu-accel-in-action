#!/usr/bin/env python3
"""从博文 Markdown 中抽取一个代码块，落成仓库里的源文件。

只做"搬运"，不改写内容：抽出来的文件与博文逐字一致，便于用 md5 对账。
改写逻辑（补 import、合并演进片段）必须人工完成，不在本工具职责内。

用法：
    # 列出某篇文章所有代码块（含序号、语言、行数、首行）
    python3 scripts/extract_block.py --article blog/010-PyCUDA归约与前缀和.md --list

    # 抽取第 22 个代码块到指定路径
    python3 scripts/extract_block.py --article blog/010-PyCUDA归约与前缀和.md \
        --index 22 --dest examples/01-gpu-programming/010-reduction-prefix-sum/reduction_scan.py

    # 不指定序号时，自动选最长的 python 块
    python3 scripts/extract_block.py --article blog/010-xxx.md --dest ...
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

BLOCK_RE = re.compile(r"```([A-Za-z0-9_+\-]*)[ \t]*\n(.*?)```", re.S)
CODE_LANGS = {"python", "cpp", "cuda", "c", "cuh"}
EXT = {"python": ".py", "cpp": ".cu", "cuda": ".cu", "c": ".c", "cuh": ".cuh"}


def load_blocks(article: Path) -> list[tuple[str, str]]:
    text = article.read_text(encoding="utf-8")
    return [(lang.lower(), body) for lang, body in BLOCK_RE.findall(text)]


def list_blocks(blocks: list[tuple[str, str]]) -> None:
    for i, (lang, body) in enumerate(blocks):
        first = body.strip().splitlines()[0][:58] if body.strip() else ""
        mark = "*" if lang in CODE_LANGS else " "
        print(f"{mark}[{i:2d}] {lang or '-':7s} {body.count(chr(10)):4d}行 | {first}")


def pick_default(blocks: list[tuple[str, str]]) -> int:
    """自动选择：最长的可编译代码块。"""
    best, best_len = -1, -1
    for i, (lang, body) in enumerate(blocks):
        if lang not in CODE_LANGS:
            continue
        length = body.count("\n")
        if length > best_len:
            best, best_len = i, length
    if best < 0:
        raise SystemExit("没有找到 python/cpp/cuda 代码块")
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article", required=True, type=Path)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if not args.article.is_file():
        raise SystemExit(f"找不到文章：{args.article}")

    blocks = load_blocks(args.article)
    if args.list or args.dest is None:
        list_blocks(blocks)
        return 0

    index = args.index if args.index is not None else pick_default(blocks)
    if not 0 <= index < len(blocks):
        raise SystemExit(f"序号越界：{index}（共 {len(blocks)} 块）")

    lang, body = blocks[index]
    if args.dest.suffix == "":
        args.dest = args.dest.with_suffix(EXT.get(lang, ".txt"))

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    # 保留末尾换行，保证 POSIX 文本文件约定
    args.dest.write_text(body.rstrip("\n") + "\n", encoding="utf-8")

    digest = hashlib.md5(args.dest.read_bytes()).hexdigest()
    print(f"已抽取 block #{index} [{lang}] -> {args.dest}")
    print(f"行数 {body.count(chr(10))} | md5 {digest}")
    print(f"来源 {args.article.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
