#!/usr/bin/env python3
"""生成 docs/reference/verification-status.md：已落地篇目目录的「实测覆盖看板」。

数据全部来自仓库自身，不手工维护：

  状态       —— 各篇 expected_output.md 首段的 **状态：…**
  证据       —— 该篇 expected_output.md 的相对路径
  分章       —— scripts/corpus.py 的 CHAPTERS

这张表存在的理由：「能跑」是本仓库的第一卖点，但实测证据散在 50 多个
expected_output.md 里，没人能一眼看出覆盖率。把覆盖率变成自动生成的一行数字，
它才能被跟踪、被要求、被写进 README 徽章。

约定：状态只认 expected_output.md 里的声明，脚本不做任何推断。
一个篇目的代码跑通了但没写进 expected_output.md，在表里仍然是「未实测」——
这正是要的效果：证据不落盘，就不算数。

用法：
    python3 scripts/gen_verify.py            # 写 docs/reference/verification-status.md
    python3 scripts/gen_verify.py --check    # 只比对，过期则退出码 1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFY = REPO_ROOT / "docs" / "reference" / "verification-status.md"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from corpus import CHAPTERS, collect  # noqa: E402  （同目录工具，复用章节词表）

STATUS_ORDER = ["已实测", "部分实测", "未实测", "不适用", "未声明"]


def classify(status: str | None, has_code: bool) -> str:
    """把 expected_output.md 的声明归到五类之一。不做推断，只做归类。

    「已实测」与「未实测」同时出现 → 部分实测：这是 011 / 019 / 077 这类
    一份代码里一半能在 CPU 上跑、一半要 N 卡的情况，把它算成「已实测」
    会把覆盖率说高，算成「未实测」又抹掉已有证据。
    """
    if not has_code:
        return "不适用"
    if not status:
        return "未声明"
    tested = "已实测" in status
    untested = "未实测" in status
    if tested and untested:
        return "部分实测"
    if tested:
        return "已实测"
    if untested:
        return "未实测"
    if "不适用" in status:
        return "不适用"
    return "未声明"


def status_text(row: dict) -> str | None:
    """expected_output.md 里 `**状态：…**` 的完整原文；没有则 None。"""
    repo_dir = row["repo_dir"]
    if not repo_dir:
        return None
    path = REPO_ROOT / repo_dir / "expected_output.md"
    if not path.is_file():
        return None
    m = re.search(r"\*\*状态：(.+?)\*\*", path.read_text(encoding="utf-8"))
    if not m:
        return None
    return m.group(1).strip().rstrip("。").replace("`", "")


def raw_status(row: dict) -> str:
    """表格展示用：原文过长时截断（完整内容看证据文件）。"""
    text = status_text(row)
    if text is None:
        return "—"
    return text if len(text) <= 30 else text[:29] + "…"


def render() -> str:
    rows = collect()
    with_code = [r for r in rows if r["repo_dir"]]
    without_code = [r for r in rows if not r["repo_dir"]]

    for r in with_code:
        # 用 raw_status 的原文分类，而不是 corpus.read_status 的归一化结果：
        # 后者把「已实测 + 未实测」压成「已实测」，会吃掉「部分实测」这一档。
        r["bucket"] = classify(status_text(r), has_code=bool(r["code_files"]))
    counts = {k: sum(1 for r in with_code if r["bucket"] == k) for k in STATUS_ORDER}
    total = len(with_code)
    # 分母是所有占用了篇目目录的篇目（含 039 这类无代码目录），
    # 这样「不适用」不会凭空扩大分母，也不会被算成通过。
    full = counts["已实测"]
    part = counts["部分实测"]
    pct = full / total * 100 if total else 0.0
    pct_any = (full + part) / total * 100 if total else 0.0

    out = [
        "# 实测覆盖看板（自动生成，勿手改）",
        "",
        "由 `python3 scripts/gen_verify.py` 从各篇 `expected_output.md` 的 `**状态：…**`",
        "汇总而成。**状态只认文件里的声明**：代码跑通了但没写进 `expected_output.md`，",
        "表里仍是「未实测」——证据不落盘就不算数。",
        "",
        "## 覆盖率",
        "",
        f"- 已落地篇目目录：**{total}** 篇（全书 {len(rows)} 篇，其余 {len(without_code)} 篇为片段型或纯文字篇）",
        f"- 全量已实测：**{full}** 篇（**{pct:.0f}%**）",
        f"- 部分已实测：**{part}** 篇（一份代码里只有主机端那半跑过）",
        f"- 未实测：**{counts['未实测']}** 篇",
        f"- 不适用（目录里无代码）：**{counts['不适用']}** 篇",
        f"- 未声明状态：**{counts['未声明']}** 篇" + (" ← 需补 `**状态：**` 首行" if counts["未声明"] else ""),
        "",
        f"**两个口径要分开看**：只算全量是通过率 **{pct:.0f}%**；把「部分实测」也算上则到"
        f" **{pct_any:.0f}%**。对外只应引用前者——分母固定，所以它只会因为真跑通一篇并写入证据而上升，",
        "不会因为新增概念篇而下降。",
        "",
        "## 分章",
        "",
        "| 章 | 章名 | 篇号 | 有目录 | 全量实测 | 部分实测 | 全量通过率 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for idx, name, _slug, lo, hi in CHAPTERS:
        in_ch = [r for r in with_code if r["chapter"] == idx]
        done = sum(1 for r in in_ch if r["bucket"] == "已实测")
        partial = sum(1 for r in in_ch if r["bucket"] == "部分实测")
        rate = f"{done / len(in_ch) * 100:.0f}%" if in_ch else "—"
        out.append(f"| {idx} | {name} | {lo:03d}–{hi:03d} | {len(in_ch)} | {done} | {partial} | {rate} |")

    out += [
        "",
        "## 逐篇明细",
        "",
        "| 篇 | 标题 | 章 | 状态 | 声明原文 | 证据 |",
        "|---:|---|---:|---|---|---|",
    ]
    for r in with_code:
        evidence = f"`{r['repo_dir']}/expected_output.md`"
        title = r["title"] if len(r["title"]) <= 24 else r["title"][:23] + "…"
        out.append(
            f"| {r['id']:03d} | {title} | {r['chapter']} | {r['bucket']} | {raw_status(r)} | {evidence} |"
        )

    out += ["", "## 未入库的篇目", "",
            f"以下 {len(without_code)} 篇没有独立可运行代码（判定理由见 README「剩下 28 篇为什么不收」）：", ""]
    line = "、".join(f"{r['id']:03d}" for r in without_code)
    out.append(line)

    out += [
        "",
        "## 怎么把一篇从「未实测」推到「已实测」",
        "",
        "1. 在 N 卡机器上按该篇 `README.md` 的「运行」小节执行；",
        "2. 把真实输出填进 `expected_output.md`，首行写 `**状态：已实测（硬件，日期）**`；",
        "3. 同时写清「这份证据不能说明什么」（例如合成数据、单卡、未覆盖 OOM 路径）；",
        "4. `make check` —— 本表会在 `--check` 阶段随之更新，忘记重跑会直接失败。",
        "",
        "只改 `expected_output.md` 不改代码，是这个仓库唯一允许的「升级」方式：",
        "代码必须与博文逐字一致（见 `docs/maintaining/provenance.json` 与 `scripts/build_from_blocks.py`）。",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只比对，不写文件")
    args = parser.parse_args()

    text = render()
    if args.check:
        if not VERIFY.is_file():
            print(f"缺少 {VERIFY.relative_to(REPO_ROOT)}，先运行 gen_verify.py")
            return 1
        if VERIFY.read_text(encoding="utf-8") != text:
            print("docs/reference/verification-status.md 已过期，请重新生成：python3 scripts/gen_verify.py")
            return 1
        print("docs/reference/verification-status.md 与仓库一致。")
        return 0

    VERIFY.write_text(text, encoding="utf-8")
    print(f"已写入 {VERIFY.relative_to(REPO_ROOT)}（{text.count(chr(10)) - 1} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
