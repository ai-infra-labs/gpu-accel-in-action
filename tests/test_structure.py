"""仓库结构与进度表的一致性测试。

不跑任何 CUDA，也不依赖第三方包：只用标准库检查“文档说的”和“目录里有的”是否一致。
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / "docs" / "reference" / "article-index.md"

# 只有「三位篇号-短名」目录属于 81 篇体系。ops/（教学单元）与 projects/
# （工程示例）不参与篇目结构与进度表校验——它们的溯源契约由
# docs/maintaining/provenance.json 的 standalone 段单独管（见 test_provenance.py）。
ARTICLE_DIRS = sorted(
    p for p in (REPO / "examples").glob("[0-9][0-9]-*/*")
    if p.is_dir() and re.match(r"^\d{3}-", p.name)
)
CODE_SUFFIXES = {".py", ".cu", ".c", ".cuh", ".cpp"}


def has_code(path: Path) -> bool:
    return any(f.suffix in CODE_SUFFIXES for f in path.iterdir())


def index_rows() -> list[tuple[int, str]]:
    """解析进度表的每一行。列序：篇号 | 标题 | 代码块 | 最大块 | 状态 | 目录，
    状态是第 5 列，不能用末列（末列是目录）。"""
    rows = []
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or not cells[0].isdigit():
            continue
        rows.append((int(cells[0]), cells[4]))
    return rows


class TestStructure(unittest.TestCase):
    def test_index_lists_all_81_articles(self):
        rows = index_rows()
        self.assertEqual(len(rows), 81, "进度表应恰好 81 行")
        self.assertEqual([n for n, _ in rows], list(range(1, 82)), "篇号应连续且升序")

    def test_every_dir_has_docs_and_a_row(self):
        rows = dict(index_rows())
        for path in ARTICLE_DIRS:
            num = int(path.name[:3])
            with self.subTest(article=path.name):
                self.assertIn(num, rows, f"{path.name} 未登记进 article-index.md")
                self.assertTrue(rows[num].startswith("已入库"), f"{path.name} 已存在但状态是 {rows[num]}")
                self.assertTrue((path / "README.md").is_file(), "缺 README.md")
                if not has_code(path):
                    # 无独立代码的篇目只留 README，且必须写明为什么没有
                    readme = (path / "README.md").read_text(encoding="utf-8")
                    self.assertTrue(
                        "不适用" in readme or "没有可独立运行的代码" in readme,
                        "无代码目录的 README 未说明原因",
                    )
                    continue
                expected = path / "expected_output.md"
                self.assertTrue(expected.is_file(), "缺 expected_output.md")
                self.assertGreater(len(expected.read_text(encoding="utf-8")), 80, "expected_output.md 太短，疑似占位")

    def test_readme_has_provenance_section(self):
        for path in ARTICLE_DIRS:
            readme = (path / "README.md").read_text(encoding="utf-8")
            with self.subTest(article=path.name):
                self.assertIn("## 出处", readme, "README 缺「出处」小节")
                self.assertIn("../../blog/", readme, "README 未链接到 blog/ 下的博文")

    def test_index_statistics_match_rows(self):
        counts = {"待提取": 0, "待拼装": 0, "待评估": 0, "已入库": 0}
        for _, status in index_rows():
            for key in counts:
                if status.startswith(key):
                    counts[key] += 1
                    break
            else:
                self.fail(f"未知状态：{status}")
        self.assertEqual(sum(counts.values()), 81)

        line = next(l for l in INDEX.read_text(encoding="utf-8").splitlines() if l.startswith("统计："))
        claimed = {k: int(v) for v, k in re.findall(r"(\d+)\s*(待提取|待拼装|待评估|已入库)", line)}
        self.assertEqual(claimed, counts, f"统计行与表内状态不符：{line}")

    def test_readme_follows_template(self):
        """有代码的篇目七段；无代码的篇目只要求「前置」与「出处」。"""
        common = ["## 前置", "## 出处"]
        with_code = ["## 文件", "## 运行", "## 环境要求", "## 观察点", "## 验证状态"]
        for path in ARTICLE_DIRS:
            text = (path / "README.md").read_text(encoding="utf-8")
            required = common + (with_code if has_code(path) else [])
            for section in required:
                with self.subTest(article=path.name, section=section):
                    self.assertIn(section, text, f"README 缺「{section}」小节")

    def test_expected_output_declares_status(self):
        for path in ARTICLE_DIRS:
            if not has_code(path):
                continue
            text = (path / "expected_output.md").read_text(encoding="utf-8")
            with self.subTest(article=path.name):
                self.assertTrue(
                    "已实测" in text or "未实测" in text or "不适用" in text,
                    "expected_output.md 未声明验证状态",
                )

    def test_readme_coverage_matches_verify_board(self):
        """README 的覆盖率徽章必须与 docs/reference/verification-status.md 一致。

        徽章是手写的、看板是生成的，两者不对齐时读者看到的数字就是旧的——
        这正是本仓库最不能出错的一个数。"""
        board = (REPO / "docs" / "reference" / "verification-status.md").read_text(encoding="utf-8")
        total = re.search(r"已落地篇目目录：\*\*(\d+)\*\* 篇", board)
        done = re.search(r"全量已实测：\*\*(\d+)\*\* 篇（\*\*(\d+)%\*\*）", board)
        self.assertIsNotNone(total, "VERIFY.md 缺「已落地篇目目录」行")
        self.assertIsNotNone(done, "VERIFY.md 缺「全量已实测」行")

        badge = f"{done.group(1)} / {total.group(1)} 篇全量跑通（{done.group(2)}%）"
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        with self.subTest(badge=badge):
            self.assertIn(badge, readme, "README 徽章与 docs/reference/verification-status.md 不一致，请同步")

    def test_license_paths_exist(self):
        """许可文件里点名的目录与文件必须真实存在。

        授权范围是最容易过期的一类文档：目录一搬，「01-…09-」还留在文件里，
        读者按它去找代码会扑空，法律上也让人误判边界。这里把路径当断言来守。
        """
        tokens = set()
        for name in ("LICENSE", "docs/LICENSE.md"):
            text = (REPO / name).read_text(encoding="utf-8")
            for m in re.finditer(r"`([^`]+)`", text):
                tok = m.group(1).strip()
                if not tok or " " in tok or any(ch in tok for ch in "*<>…"):
                    continue
                if "/" not in tok and "." not in tok:
                    continue  # 不是路径（如 `nvcc`、`md5`）
                tokens.add(tok)
        self.assertTrue(tokens, "许可文件里没有任何路径，检查解析是否失效")
        missing = [t for t in sorted(tokens) if not (REPO / t.rstrip("/")).exists()]
        self.assertEqual(missing, [], f"许可文件指向不存在的路径：{missing}")

    def test_notice_lists_third_party_assets(self):
        """NOTICE 里登记的第三方图必须真实存在，且不能把作者自制的图算进去。"""
        notice = REPO / "blog" / "assets" / "NOTICE.md"
        self.assertTrue(notice.is_file(), "缺 blog/assets/NOTICE.md")
        text = notice.read_text(encoding="utf-8")
        listed = re.findall(r"\| `([^`]+\.(?:jpg|png))` \|", text)
        self.assertTrue(listed, "NOTICE 未列出任何第三方图")
        for name in listed:
            with self.subTest(asset=name):
                self.assertTrue((REPO / "blog" / "assets" / name).is_file(), "登记了不存在的图")

    def test_every_chapter_has_index_readme(self):
        """每章目录必须有 README，且列出该章全部篇号（含未入库的片段篇）。

        章节目录只放「有代码的篇目」，于是第 4、6 章可能一个可运行目录都没有。
        章 README 就是给这种情况兜底的导航：读者按章进来，至少能看到本章的全部篇目。
        """
        sys.path.insert(0, str(REPO / "scripts"))
        from corpus import CHAPTERS  # noqa: PLC0415  （同仓库工具脚本，延迟导入避免污染）

        ranges = {slug: (lo, hi) for _idx, _name, slug, lo, hi in CHAPTERS}
        for chapter_dir in sorted(REPO.glob("[0-9][0-9]-*")):
            if not chapter_dir.is_dir() or chapter_dir.name not in ranges:
                continue
            lo, hi = ranges[chapter_dir.name]
            readme = chapter_dir / "README.md"
            with self.subTest(chapter=chapter_dir.name):
                self.assertTrue(readme.is_file(), f"{chapter_dir.name} 缺章索引 README.md")
                text = readme.read_text(encoding="utf-8")
                missing = [n for n in range(lo, hi + 1) if f"{n:03d}" not in text]
                if missing:
                    self.fail(
                        f"{chapter_dir.name}/README.md 漏列 "
                        f"{', '.join(f'{n:03d}' for n in missing)}"
                    )


if __name__ == "__main__":
    unittest.main()
