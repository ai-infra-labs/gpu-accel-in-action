#!/usr/bin/env python3
"""生成九章目录的索引 README（`01-gpu-programming/README.md` … `09-medical-imaging/README.md`）。

为什么需要它：章节目录只放「有独立代码的篇目」，第 4、6 章这种一章只有 1 个目录、
甚至那 1 个还没有代码的情况，会让读者误以为「这章没内容」。章 README 把本章
**全部**篇目（含只有讲解片段的）列出来，导航才完整。

数据来源：

  篇号/标题/章节  —— scripts/corpus.py（扫 blog/ 得出，唯一来源）
  是否有代码       —— 篇目目录是否存在
  实测状态         —— 各篇 expected_output.md 的 **状态：…**（复用 gen_verify 的归类）
  本章目标/必读/误解 —— 下方 CHAPTER_NOTES，从 docs/learning-path.md 收敛而来

用法：
    python3 scripts/gen_chapters.py            # 写 9 个章 README
    python3 scripts/gen_chapters.py --check    # 只比对，过期则退出码 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from corpus import CHAPTERS, collect  # noqa: E402
from gen_verify import classify, status_text  # noqa: E402  （复用同一套实测归类）

# 每章的目标 / 必读 / 常见误解，措辞与 docs/learning-path.md 保持一致。
CHAPTER_NOTES = {
    1: ("GPU 里没有矩阵，只有线程编号和一段连续显存。这一章把「我怎么知道第 i 个线程该算哪个数」"
        "变成不需要想的事。",
        [1, 10, 17],
        "以为线程越多越快。006 的 `grid=(7,7)` 会多启动 144 个线程靠边界判断挡掉——多出来的线程不是免费的。"),
    2: ("把「参数少了」「FLOPs 降了」「文件小了」「跑得快了」这四件事分开算。",
        [23, 26],
        "以为量化就是「把 float 换成 int8 就能加速」。021 会告诉你，反量化之后卷积仍是 fp32，"
        "省的是显存与带宽，不是乘加次数。"),
    3: ("一次请求的时间不是一个数。模型执行、数据搬运、框架调度、排队，要分得开才谈得上优化。",
        [28, 29],
        "把「引擎执行时间」当端到端时延。两者之差就是拷贝与调度，不拆开就永远在优化错的东西。"),
    4: ("摄像头为什么「认出人几十毫秒，提醒却晚两秒」——端侧的瓶颈常常是链路，不是模型。",
        [39],
        "以为设备上有 NPU，模型就会跑在 NPU 上。执行提供者（EP）注册了，不等于你的算子被它接管，"
        "要读图切分报告。"),
    5: ("并发是「有机会重叠」，不是「一定会加速」。这一章把 DP / PP / TP 三种切法落到 rank 编号与张量形状上。",
        [47, 48, 51],
        "把 TP 的「部分和相加」当成 DP 的「梯度平均」，在同一段代码里混用两种语义。"),
    6: ("什么时候值得自己写 Kernel。答案不是「需要更快」，而是「现有实现多做了你不需要的事」。",
        [56],
        "以为手写一定更快。056 的三路对照里，小 N 时 eager / compile / 手写打平——手写买到的是可控，不是倍数。"),
    7: ("六个可独立验证的机制：显存记账、合并访存、共享内存容量、常量内存广播、页锁定内存、异步重叠。",
        [61, 62, 63, 65],
        "以为「加了异步」就一定重叠。065 的判据是总时间接近 `max(拷贝, 计算)`，不是代码里出现了 stream。"),
    8: ("一条能核对结果的推理链路：导出 → 构建 → 运行 → 计时 → 量化。",
        [67, 69, 70, 73],
        "以为「导出成功」等于「数值一致」。067 要用 13 组用例才敢说后者。"),
    9: ("三维体积装不进显存时，怎么切、怎么拼、怎么评估。",
        [77, 81],
        "以为评估不用验证。081 里 Dice 与手算逐位一致才算数，空掩膜记 `null` 而不是 1.0。"
        "074/078/081 用的是**合成数据**，验证的是几何与公式，不是病例。"),
}


def render_chapter(idx: int, name: str, slug: str, lo: int, hi: int, rows: list[dict]) -> str:
    in_ch = [r for r in rows if lo <= r["id"] <= hi]
    with_code = [r for r in in_ch if r["repo_dir"]]
    full = [r for r in with_code if classify(status_text(r), bool(r["code_files"])) == "已实测"]
    goal, must_read, misconception = CHAPTER_NOTES[idx]

    out = [
        f"# 第 {idx} 章 · {name}",
        "",
        f"> 篇号 {lo:03d}–{hi:03d} · 本章 {len(in_ch)} 篇 · 有独立代码 {len(with_code)} 篇 · 全量已实测 {len(full)} 篇",
        "",
        f"**本章目标**：{goal}",
        "",
        "**必读**：" + " → ".join(f"{n:03d}" for n in must_read),
        "",
        f"**最常见的误解**：{misconception}",
        "",
        "## 篇目",
        "",
        "| 篇 | 标题 | 代码 | 实测 | 位置 |",
        "|---:|---|---|---|---|",
    ]
    for r in in_ch:
        num = r["id"]
        title = r["title"].replace("|", "\\|")
        if r["repo_dir"]:
            has_code = f"{len(r['code_files'])} 个文件" if r["code_files"] else "无（仅 README）"
            status = classify(status_text(r), bool(r["code_files"]))
            where = f"[`{Path(r['repo_dir']).name}/`]({Path(r['repo_dir']).name}/)"
        else:
            has_code = "无"
            status = "—"
            where = "仅博文"
        out.append(f"| {num:03d} | {title} | {has_code} | {status} | {where} |")

    out += [
        "",
        "## 说明",
        "",
        f"本目录只放「有独立代码的篇目」（{len(with_code)} / {len(in_ch)} 篇）。"
        "其余篇目的代码是讲解片段，拼成文件就等于替作者写新代码，因此不单独入库——"
        "读 [`blog/`](../../blog/) 里的博文即可，判定标准见"
        "[仓库 README 的「剩下 28 篇为什么不收」](../../README.md)。",
        "",
        "标「仅博文」的行不是待办，是**判定过的不入库**。",
        "",
        f"章内依赖顺序与全书路线见 [`docs/learning-path.md`](../../docs/learning-path.md)；"
        "逐篇实测证据见 [`docs/reference/verification-status.md`](../../docs/reference/verification-status.md)。",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只比对，不写文件")
    args = parser.parse_args()

    rows = collect()
    stale, missing = [], []
    for idx, name, slug, lo, hi in CHAPTERS:
        path = REPO_ROOT / "examples" / slug / "README.md"
        text = render_chapter(idx, name, slug, lo, hi, rows)
        if not path.parent.is_dir():
            continue
        if args.check:
            if not path.is_file():
                missing.append(str(path.relative_to(REPO_ROOT)))
            elif path.read_text(encoding="utf-8") != text:
                stale.append(str(path.relative_to(REPO_ROOT)))
        else:
            path.write_text(text, encoding="utf-8")

    if args.check:
        if missing or stale:
            for item in missing:
                print(f"缺少 {item}")
            for item in stale:
                print(f"已过期 {item}")
            print("请重新生成：python3 scripts/gen_chapters.py")
            return 1
        print("九章 README 与仓库一致。")
        return 0

    print(f"已写入 {len(CHAPTERS)} 个章索引 README")
    return 0


if __name__ == "__main__":
    sys.exit(main())
