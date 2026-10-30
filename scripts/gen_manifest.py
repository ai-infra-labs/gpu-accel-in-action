#!/usr/bin/env python3
"""生成 docs/_data/articles.json：全仓库语料的机器可读清单。

清单是「检索快」的基石——search.py 直接读它，不每次扫 81 篇 Markdown。
它也是 ARTICLE_INDEX.md 的生成源（见 gen_index.py），一处更新、处处一致。

入库状态（ingest_status）是唯一需要人工策展的字段，首次从现有
docs/reference/article-index.md 的「状态」列引导；之后新增篇目时直接改这份 JSON。

用法：
    python3 scripts/gen_manifest.py            # 写 docs/_data/articles.json
    python3 scripts/gen_manifest.py --check    # 只比对，过期则退出码 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import REPO_ROOT, ARTICLE_INDEX, collect  # noqa: E402

MANIFEST = REPO_ROOT / "docs" / "_data" / "articles.json"
ROW_RE = re.compile(r"^\| (\d{3}) \| .+? \| \d+ \| \d+ \| (.+?) \| .+? \|$")


def bootstrap_ingest_status() -> dict[int, str]:
    """从现有 ARTICLE_INDEX.md 的「状态」列取策展的入库状态。"""
    text = ARTICLE_INDEX.read_text(encoding="utf-8")
    out: dict[int, str] = {}
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return out


def render() -> str:
    rows = collect(bootstrap_ingest_status())
    payload = {
        "meta": {
            "source": "scripts/gen_manifest.py",
            "count": len(rows),
            "note": "ingest_status 为策展字段；其余字段由仓库自动推导。",
        },
        "articles": rows,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只比对，不写文件")
    args = parser.parse_args()

    text = render()
    if args.check:
        if not MANIFEST.is_file():
            print(f"缺少 {MANIFEST.relative_to(REPO_ROOT)}，先运行 gen_manifest.py")
            return 1
        if MANIFEST.read_text(encoding="utf-8") != text:
            print("docs/_data/articles.json 已过期，请重新生成：python3 scripts/gen_manifest.py")
            return 1
        print("docs/_data/articles.json 与仓库一致。")
        return 0

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(text, encoding="utf-8")
    print(f"已写入 {MANIFEST.relative_to(REPO_ROOT)}（{len(json.loads(text)['articles'])} 篇）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
