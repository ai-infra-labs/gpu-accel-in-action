#!/usr/bin/env python3
"""校验（或重建）仓库代码与博文代码块的对应关系。

三种来源：

  1. **逐字抽取**：仓库文件与某个博文代码块 md5 完全相同，不需要登记。
  2. **拼接**：一篇的代码被拆成几块讲解，单块缺 import 或常量，
     必须按顺序拼起来才能运行。这类文件登记在 `docs/maintaining/provenance.json`，
     由本脚本按块序号重建并比对。
  3. **独立项目（standalone）**：自带构建与运行入口的完整工程（`projects/`），
    按目录前缀登记在 `PROVENANCE.json` 的 `standalone` 段。这类代码的合格标准
    是「自己能构建、能跑出结果」，不是「能不能对上某篇博文的代码块」——
    两件事本来无关，混在一条规则里会让「还没写文章」看起来像「代码有问题」。

`--strict` 因此只对第 3 类提要求：登记里声明的构建入口必须真实存在。

拼接规则固定为：各块去掉首尾空行后，用**一个空行**连接，末尾补一个换行。
规则是唯一的，所以"仓库文件 = 博文块的有序拼接"可以被机器验证。

用法：
    python3 scripts/build_from_blocks.py --check     # 只比对，不写文件
    python3 scripts/build_from_blocks.py --check --strict   # 有待博文即失败
    python3 scripts/build_from_blocks.py --write     # 按登记重建拼接文件
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROVENANCE = REPO_ROOT / "docs" / "maintaining" / "provenance.json"
BLOG_DIR = REPO_ROOT / "blog"

BLOCK_RE = re.compile(r"```([A-Za-z0-9_+\-]*)[ \t]*\n(.*?)```", re.S)
CODE_LANGS = {"python", "cpp", "cuda", "c", "cuh"}
CODE_SUFFIXES = {".py", ".cu", ".c", ".cuh", ".cpp"}


def load_blocks(article: Path) -> list[tuple[str, str]]:
    """返回文章的全部代码围栏，序号与 extract_block.py 的 --index 一致。"""
    text = article.read_text(encoding="utf-8")
    return [(lang.lower(), body) for lang, body in BLOCK_RE.findall(text)]


def assemble(parts: list[str]) -> str:
    """拼接规则：各块去首尾空行，空行相连，末尾补换行。"""
    return "\n\n".join(part.strip("\n") for part in parts) + "\n"


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def repo_files() -> list[Path]:
    """仓库里所有需要溯源判定的代码文件。

    除工具目录（scripts/tests/cuda_kernels）与博文目录（blog）外，一律纳入：
    篇目目录、教学单元 ops/、工程示例 projects/ 都跑不掉。任何新代码要么能对上
    博文块，要么有拼接登记，要么在 pending 里——没有第四条路。

    cuda_kernels/ 是工具而非篇目代码：它一行源码都不复制，只按路径加载章节目录
    里的规范文件（见 docs/reference/package.md）。把它排除，是为了让这条契约仍然只有
    「章节目录」一个入口。
    """
    files = []
    for path in REPO_ROOT.rglob("*"):
        if path.suffix not in CODE_SUFFIXES:
            continue
        rel = path.relative_to(REPO_ROOT)
        if rel.parts[0] in {"scripts", "tests", "blog", "docs", "cuda_kernels", ".git"}:
            continue
        if ".git" in rel.parts or "build" in rel.parts:
            continue
        files.append(path)
    return sorted(files)


def standalone_prefixes(data: dict) -> list[str]:
    """独立项目的目录前缀列表（见 provenance.json 的 standalone 段）。"""
    return [p for entry in data.get("standalone", []) for p in entry.get("paths", [])]


def is_standalone(rel: str, prefixes: list[str]) -> bool:
    return any(rel.startswith(prefix) for prefix in prefixes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="按登记重建拼接文件")
    parser.add_argument("--check", action="store_true", help="只比对（默认行为）")
    parser.add_argument("--strict", action="store_true",
                        help="额外校验独立项目声明的构建入口是否真实存在")
    args = parser.parse_args()

    if not PROVENANCE.is_file():
        raise SystemExit(f"缺少 {PROVENANCE}")

    data = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    registry = data["files"]
    prefixes = standalone_prefixes(data)

    # 博文所有块：md5 -> (文章名, 序号)
    block_index: dict[str, tuple[str, int]] = {}
    for article in sorted(BLOG_DIR.glob("*.md")):
        for i, (lang, body) in enumerate(load_blocks(article)):
            if lang not in CODE_LANGS:
                continue
            text = body.rstrip("\n") + "\n"
            block_index.setdefault(md5(text), (article.name, i))

    def render(entry: dict) -> str:
        article = BLOG_DIR / entry["article"]
        if not article.is_file():
            raise SystemExit(f"找不到博文：{article}")
        blocks = load_blocks(article)
        try:
            parts = [blocks[i][1] for i in entry["blocks"]]
        except IndexError:
            raise SystemExit(f"块序号越界：{entry['article']} {entry['blocks']}")
        return assemble(parts)

    rebuilt: list[str] = []
    verbatim = 0
    standalone_hits = 0
    mismatched: list[str] = []

    if args.write:
        for rel, entry in registry.items():
            path = REPO_ROOT / rel
            expected = render(entry)
            old = path.read_text(encoding="utf-8") if path.is_file() else None
            if old != expected:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(expected, encoding="utf-8")
                rebuilt.append(rel)

    for path in repo_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")

        if md5(text) in block_index:
            verbatim += 1
            continue

        entry = registry.get(rel)
        if entry is not None:
            if md5(render(entry)) != md5(text):
                mismatched.append(f"{rel}: 与拼接结果不一致（块 {entry['blocks']}）")
            continue

        if is_standalone(rel, prefixes):
            standalone_hits += 1
            continue

        mismatched.append(f"{rel}: 既不是任何博文块，也没有拼接或待博文登记")

    # standalone 登记必须真能对上文件，否则豁免会悄悄失效
    all_rels = [p.relative_to(REPO_ROOT).as_posix() for p in repo_files()]
    for prefix in prefixes:
        if not any(rel.startswith(prefix) for rel in all_rels):
            mismatched.append(f"{prefix}: 独立项目登记的前缀下没有任何代码文件")

    if not args.write:
        for rel in registry:
            if not (REPO_ROOT / rel).is_file():
                mismatched.append(f"{rel}: 已登记但文件不存在")

    print(f"逐字一致：{verbatim} 个文件")
    print(f"拼接登记：{len(registry)} 个文件")
    print(f"独立项目（standalone）：{standalone_hits} 个文件 / {len(prefixes)} 条目录登记（不计入博文溯源）")
    if args.write:
        print(f"已重建：{len(rebuilt)} 个 -> " + ", ".join(rebuilt or ["（无变化）"]))
    if mismatched:
        print(f"\n不一致 {len(mismatched)} 条：")
        for item in mismatched:
            print("  x " + item)
        return 1
    if args.strict:
        for entry in data.get("standalone", []):
            built = entry.get("built_by", "")
            if not built:
                mismatched.append(f"{entry.get('paths')}: --strict 要求声明 built_by 构建入口")
            elif not (REPO_ROOT / built).is_file():
                mismatched.append(f"{built}: 声明的构建入口不存在")
        if mismatched:
            for item in mismatched:
                print("  x " + item)
            return 1
    print("\n篇目代码全部可追溯到博文代码块；独立项目按自身构建标准验收。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
