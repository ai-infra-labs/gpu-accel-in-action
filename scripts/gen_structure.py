#!/usr/bin/env python3
"""生成 docs/maintaining/repository-layout.md：仓库目录结构说明书。

为什么要生成而不是手写：目录会增删、篇目数会变、脚本会加。手写的结构文档
两周后就开始骗人——读者按图索骥找不到文件，比没有文档更糟。

数据来源全部是仓库自身：

    顶层目录 / 篇目数 / 文件数  —— 扫磁盘
    脚本与测试的一句话说明      —— 各文件模块 docstring 的首句
    章节划分                 —— scripts/corpus.py 的 CHAPTERS
    派生文档的生成者           —— scripts/GENERATED 表（本文件内维护，只有 6 条）

只有「目录是干什么的」这类判断需要人写，收在 DIR_NOTES 里。

用法：
    python3 scripts/gen_structure.py            # 写 docs/maintaining/repository-layout.md
    python3 scripts/gen_structure.py --check    # 只比对，过期则退出码 1
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "docs" / "maintaining" / "repository-layout.md"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from corpus import CHAPTERS  # noqa: E402

CODE_SUFFIXES = {".py", ".cu", ".c", ".cuh", ".cpp"}

# 顶层目录：作用 / 能否手改 / 谁生成
DIR_NOTES = {
    "blog": ("81 篇博文的唯一事实源，文件名 `NNN-短名.md`；配图在 `blog/assets/`",
             "可以，但改代码块必须同步 examples/ 下的代码并重跑 `make check`", "手写"),
    "examples": ("九章配套代码：`<章>/<篇号-短名>/`，章内只放有代码目录的篇目",
                 "可以，新增篇目走 CONTRIBUTING 的流程", "混合（章索引 README 自动生成）"),
    "projects": ("可独立构建的完整工程（CT 重建线），不以是否发表过文章作为合格标准",
                 "可以，新项目须在 provenance.json 的 standalone 段登记构建入口", "手写"),
    "cuda_kernels": ("加载层：按名字找到/加载/运行某一篇的脚本，不复制任何源码",
                     "可以（改完全仓测试会校验模块名唯一）", "手写"),
    "scripts": ("工具脚本：抽取、校验、派生文档生成",
                "可以，新增脚本要登记到 `static_check.LOCAL_MODULES`（若被 import）", "手写"),
    "tests": ("结构 / 骨架 / 溯源 / 静态 / 冒烟 / 包契约，纯标准库，无卡可跑",
              "可以，新增不变量优先加在这里", "手写"),
    "requirements": ("三层依赖：CPU 可跑 / 必须 GPU / 特殊环境（TensorRT、Triton、MPI）",
                     "可以", "手写"),
    "docs": ("文档与派生看板：路线、能力矩阵、实测看板、构建、包说明、方案记录",
             "部分：标「自动生成」的不要手改", "混合"),
    ".github": ("CI：`make check`（无 GPU，只验证文档与代码对得上）",
                "可以", "手写"),
}

# 派生文档 → 生成脚本。只登记「重跑一次就能重建」的，手写文档不进这张表。
GENERATED = {
    "docs/reference/capability-matrix.md": "scripts/gen_atlas.py",
    "docs/reference/verification-status.md": "scripts/gen_verify.py",
    "docs/reference/article-index.md": "scripts/gen_index.py",
    "docs/_data/articles.json": "scripts/gen_manifest.py",
    "docs/maintaining/repository-layout.md": "scripts/gen_structure.py",
}

# 九章备注：一句话说明这一章解决什么问题（措辞与 docs/learning-path.md 一致）
CHAPTER_BLURB = {
    1: "线程→数据的映射、共享内存、同步",
    2: "参数量 / FLOPs / 显存三个量分开算",
    3: "拆开一次推理请求的时间",
    4: "端侧链路的真实约束",
    5: "DP / PP / TP 怎么切、怎么验证",
    6: "什么时候值得自己写算子",
    7: "访存与传输的六个机制",
    8: "ONNX → 引擎 → 运行时",
    9: "三维医疗影像的工程闭环",
}


def first_sentence(path: Path) -> str:
    """取模块 docstring 的首句，作为一句话说明。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return "（解析失败）"
    doc = ast.get_docstring(tree) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line.rstrip("。") if len(line) > 4 else line
    return "（无 docstring）"


def count_files(d: Path) -> int:
    return sum(1 for p in d.rglob("*") if p.suffix in CODE_SUFFIXES)


def chapter_rows() -> list[tuple[int, str, str, int, int, int, int, int]]:
    """(章, 主题, 目录, lo, hi, 篇目数, 代码目录数, 代码文件数)"""
    rows = []
    for idx, name, slug, lo, hi in CHAPTERS:
        d = REPO_ROOT / "examples" / slug
        dirs = [p for p in d.glob("[0-9][0-9][0-9]-*") if p.is_dir()] if d.is_dir() else []
        rows.append((idx, name, slug, lo, hi, hi - lo + 1, len(dirs), count_files(d)))
    return rows


def render() -> str:
    chapters = chapter_rows()
    total_articles = sum(r[5] for r in chapters)
    total_dirs = sum(r[6] for r in chapters)
    total_code = sum(r[7] for r in chapters)

    out = [
        "# 仓库结构说明（自动生成，勿手改）",
        "",
        "由 `python3 scripts/gen_structure.py` 扫描仓库生成：目录与计数来自磁盘，"
        "脚本说明来自各自 docstring 的首句，章节划分来自 `scripts/corpus.py`。",
        "改完目录或新增脚本后 `make gen`，`make check` 会在过期时失败。",
        "",
        "## 一、一层看全",
        "",
        "```",
        "gpu-accel-in-action/",
        "├── blog/            81 篇博文 + assets/ 配图（代码块的 md5 是溯源基准）",
        "├── examples/        九章配套代码：examples/01-gpu-programming/…/09-medical-imaging/",
        "├── projects/        可独立构建的完整项目（CT 重建线），按自己的构建验收",
        "├── cuda_kernels/    加载层：按名字查找 / 加载 / 运行某一篇",
        "├── scripts/         工具脚本：抽取、校验、派生文档",
        "├── tests/           结构 / 溯源 / 静态 / 冒烟 / 包契约测试",
        "├── requirements/    三层依赖（CPU / GPU / 特殊环境）",
        "├── docs/            入门 / 参考 / 维护 / 决策记录（见 docs/README.md）",
        "├── .github/         CI（无 GPU，只跑 make check）",
        "└── 根文件           README / Makefile / CMakeLists / pyproject / 贡献与变更",
        "```",
        "",
        f"篇目：**{total_articles}** 篇博文，**{total_dirs}** 个代码目录，**{total_code}** 个代码文件。",
        "",
        "## 二、顶层目录逐个说",
        "",
        "| 目录 | 作用 | 能否手改 | 性质 |",
        "|---|---|---|---|",
    ]
    for name, (purpose, editable, origin) in DIR_NOTES.items():
        out.append(f"| `{name}/` | {purpose} | {editable} | {origin} |")

    out += [
        "",
        "## 三、三个代码区，规则不同",
        "",
        "仓库里的代码分三类，**放在哪决定了它受什么约束**。混放是最常见的错误。",
        "",
        "| 位置 | 是什么 | 溯源契约 | 构建方式 |",
        "|---|---|---|---|",
        "| `examples/<章>/<NNN-短名>/` | 某篇博文的逐字代码 | md5 必须命中 `blog/` 的代码块，或登记为拼接 | 单文件直接跑（多数是 PyCUDA 脚本） |",
        "| `projects/ct-reconstruction/` | 可独立构建的完整工程 | 登记在 `docs/maintaining/provenance.json` 的 `standalone`；按自己的构建与测试验收 | `ops/` 单文件；`src|include|apps/` 走 CMake |",
        "| `cuda_kernels/` | 加载层（工具，不是篇目代码） | 排除在溯源检查外，且不得复制篇目源码 | 不需要构建 |",
        "",
        "判定口诀：**「这篇教程里有这段代码吗？」**有 → `examples/`；是一整个能独立构建的工程 → `projects/`；"
        "只是工具 → `scripts/` 或 `cuda_kernels/`。",
        "",
        "三者**互不越界**：篇目代码不许脱离教程独自演化（md5 对账），完整工程不许拿「还没写文章」自我豁免"
        "（`built_by` 必须真实存在），工具不许复制任何篇目源码。",
        "",
        "## 四、九章导航",
        "",
        "| 章 | 目录 | 篇号范围 | 主题 | 这一章解决什么 | 篇目数 | 代码目录 | 代码文件 |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]
    for idx, name, slug, lo, hi, n_art, n_dir, n_code in chapters:
        out.append(f"| {idx} | `examples/{slug}/` | {lo:03d}–{hi:03d} | {name} | "
                   f"{CHAPTER_BLURB.get(idx, '')} | {n_art} | {n_dir} | {n_code} |")

    out += [
        "",
        "## 六、docs/ 按用途分区",
        "",
        "文档按**读者要解决什么问题**分区，不按「是否自动生成」分区：",
        "",
        "| 分区 | 放什么 | 谁看 |",
        "|---|---|---|",
        "| `docs/` 根 | 入口：`README.md` 导航 + 学习路线 + 快速开始 | 第一次来的人 |",
        "| `docs/reference/` | 查阅型：文章索引、能力矩阵、验证状态、包说明 | 按需跳转的人 |",
        "| `docs/maintaining/` | 维护者：仓库结构、示例约定、溯源契约与机器清单 | 要改仓库的人 |",
        "| `docs/decisions/` | 决策记录：为什么这样组织、当时的取舍 | 半年后想改结构的人 |",
        "",
        "标「自动生成」的不要手改，跑 `make gen` 重建：",
        "",
        "| 文件 | 生成者 | 性质 |",
        "|---|---|---|",
    ]
    seen = set()
    for rel, gen in GENERATED.items():
        out.append(f"| `{rel}` | `{gen}` | 自动生成 |")
        seen.add(rel)
    for p in sorted((REPO_ROOT / "docs").rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = str(p.relative_to(REPO_ROOT))
        if rel in seen or p.suffix not in {".md", ".json"}:
            continue
        out.append(f"| `{rel}` | — | 手写 |")

    out += ["", "## 七、scripts/ 工具", "",
            "| 脚本 | 一句话 |", "|---|---|"]
    for p in sorted((REPO_ROOT / "scripts").glob("*.py")):
        out.append(f"| `{p.name}` | {first_sentence(p)} |")

    out += ["", "## 八、tests/ 测试", "",
            "全部只用标准库，无 GPU 可跑；缺第三方依赖的用例自动 skip。",
            "",
            "| 模块 | 守什么 |", "|---|---|"]
    for p in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        out.append(f"| `{p.name}` | {first_sentence(p)} |")

    out += [
        "",
        "## 九、根文件",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `README.md` | 定位、两个硬规矩、目录约定、验证链路、已收录清单 |",
        "| `Makefile` | `check` / `gen` / `test` / `strict` / `build` / `search` 六个入口 |",
        "| `CMakeLists.txt` | 单文件教学单元的构建（默认 `native` 架构，见 `docs/getting-started.md`） |",
        "| `CODE_OF_CONDUCT.md` | 贡献者行为准则（改编自 Contributor Covenant 2.1） |",
        "| `SECURITY.md` | 安全问题怎么报、哪些算安全问题、多久回应 |",
        "| `pyproject.toml` | 打包配置：只打包 `cuda_kernels`，不把 81 篇脚本收进 wheel |",
        "| `CONTRIBUTING.md` | 新增一篇 / 新增代码的流程与检查项 |",
        "| `CHANGELOG.md` | 仓库级变更记录 |",
        "| `LICENSE` | 代码 MIT |",
        "| `docs/LICENSE.md` | 文档与博文 CC BY 4.0 |",
        "| `.gitignore` | 忽略 `__pycache__`、打包产物与 `build/` |",
        "| `.github/workflows/ci.yml` | push / PR 时跑 `make check` |",
        "",
        "## 十、命名与放置约定（速查）",
        "",
        "| 规则 | 正确 | 反例 |",
        "|---|---|---|",
        "| 篇目目录 = 三位篇号 + 短名 | `011-tiled-matmul-shared-memory/` | `tiled-matmul/` |",
        "| 博文文件名 = 篇号 + 短名 | `blog/011-pycuda-tiled-matmul-….md` | `tiled-matmul.md` |",
        "| 图 = `figure-<主题>-NN` | `blog/assets/figure-conv2d-01.jpg` | `blog/assets/orca-paste-1789634491.png` |",
        "| 新增篇目代码放 examples/ | `examples/07-memory-and-transfer/061-…/` | 仓库根下 `061-…/` |",
        "| 新代码必须可溯源 | 对上博文块 / 拼接登记 / pending | 直接丢进仓库任意位置 |",
        "| 派生文档不手改 | 跑 `make gen` | 手改 `docs/reference/capability-matrix.md` |",
        "",
        "## 附：完整目录清单",
        "",
        "```",
    ]
    def walk(d: Path, depth: int, max_depth: int = 3) -> None:
        for sub in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("__")):
            n_code = count_files(sub)
            n_direct = sum(1 for x in sub.iterdir() if x.is_file())
            indent = "  " * (depth - 1)
            # 两个数分开写：direct = 本层直属文件，code = 含子孙目录的代码文件
            detail = f"直属 {n_direct} 个文件" if n_direct else "本层无文件"
            out.append(f"{indent}{sub.name}/  ({detail}；递归 {n_code} 个代码文件)")
            if depth < max_depth:
                walk(sub, depth + 1, max_depth)

    for d in sorted(p for p in REPO_ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")):
        n_code = count_files(d)
        n_direct = sum(1 for x in d.iterdir() if x.is_file())
        detail = f"直属 {n_direct} 个文件" if n_direct else "本层无文件"
        out.append(f"{d.name}/  ({detail}；递归 {n_code} 个代码文件)")
        walk(d, 1)
    out += ["```"]

    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只比对，不写文件")
    args = parser.parse_args()

    text = render()
    if args.check:
        if not OUT.is_file():
            print(f"缺少 {OUT.relative_to(REPO_ROOT)}，先运行 gen_structure.py")
            return 1
        if OUT.read_text(encoding="utf-8") != text:
            print("docs/maintaining/repository-layout.md 已过期，请重新生成：python3 scripts/gen_structure.py")
            return 1
        print("docs/maintaining/repository-layout.md 与仓库一致。")
        return 0

    OUT.write_text(text, encoding="utf-8")
    print(f"已写入 {OUT.relative_to(REPO_ROOT)}（{text.count(chr(10)) - 1} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
