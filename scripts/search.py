#!/usr/bin/env python3
"""在仓库语料中检索篇目：按 id / 标题 / 标签(mechanisms) / 章节 / 状态 / 依赖。

直接读 docs/_data/articles.json（由 gen_manifest.py 生成），不每次扫 81 篇
Markdown，所以“检索快”。零三方依赖。

用法：
    python3 scripts/search.py nccl                 # 自由文本：命中标题/标签/章节…
    python3 scripts/search.py --tag nccl           # 按机制标签精确匹配
    python3 scripts/search.py --chapter 5          # 第 5 章（并行与分布式）
    python3 scripts/search.py --status 未实测       # 按入库/验证状态过滤
    python3 scripts/search.py --id 047             # 按篇号
    python3 scripts/search.py --list               # 列出全部 81 篇
    python3 scripts/search.py nccl --json          # 机器可读输出
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import REPO_ROOT, LEGEND  # noqa: E402

MANIFEST = REPO_ROOT / "docs" / "_data" / "articles.json"


def load() -> list[dict]:
    if not MANIFEST.is_file():
        raise SystemExit("缺少 docs/_data/articles.json，先运行：python3 scripts/gen_manifest.py")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["articles"]


def matches(a: dict, args: argparse.Namespace) -> bool:
    if args.id is not None and a["id"] != args.id:
        return False
    if args.chapter is not None and a["chapter"] != args.chapter:
        return False
    if args.status and args.status not in a["ingest_status"] and args.status != a["verify_status"]:
        return False
    if args.tag and args.tag not in a["mechanisms"]:
        return False
    if args.query:
        q = args.query.lower()
        hay = " ".join([
            str(a["id"]), a["slug"], a["title"], a["chapter_name"],
            " ".join(a["mechanisms"]), " ".join(a["deps"]),
            a["ingest_status"], a["verify_status"] or "",
        ]).lower()
        if q not in hay:
            return False
    return True


def show(a: dict) -> str:
    mech = "、".join(f"{m}({LEGEND.get(m, '')})" for m in a["mechanisms"]) or "—"
    loc = a["repo_dir"] or a["blog_path"]
    return (f"{a['id']:03d}  {a['title']}\n"
            f"      章节 {a['chapter']}·{a['chapter_name']}  "
            f"状态 {a['ingest_status']}（验证 {a['verify_status'] or '—'}）\n"
            f"      标签 [{mech}]  依赖 {', '.join(a['deps']) or '—'}\n"
            f"      位置 {loc}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", nargs="?", help="自由文本（标题/标签/章节/依赖/id）")
    p.add_argument("--tag", help="按机制标签精确匹配，如 nccl / shared-memory")
    p.add_argument("--chapter", type=int, help="按章号过滤，如 5")
    p.add_argument("--status", help="按状态子串过滤，如 未实测 / 待拼装")
    p.add_argument("--id", type=int, help="按篇号精确匹配")
    p.add_argument("--list", action="store_true", help="列出全部 81 篇")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    args = p.parse_args()

    arts = [a for a in load() if args.list or matches(a, args)]
    if not arts:
        print("无匹配篇目。")
        return 0
    arts.sort(key=lambda a: a["id"])

    if args.json:
        print(json.dumps(arts, ensure_ascii=False, indent=2))
    else:
        for a in arts:
            print(show(a))
        print(f"\n共 {len(arts)} 篇。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
