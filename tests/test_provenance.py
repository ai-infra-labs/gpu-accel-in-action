"""代码与博文的溯源测试：逐字或拼接，两种来源都必须可验证。

夹具依赖两份材料：博文目录（`blog/`）与 `docs/maintaining/provenance.json`。
博文现已纳入仓库（`blog/`），所以这个测试总是运行，不再静默跳过。
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BLOG = REPO / "blog"
SCRIPT = REPO / "scripts" / "build_from_blocks.py"
BLOCK_RE = re.compile(r"```([A-Za-z0-9_+\-]*)[ \t]*\n(.*?)```", re.S)


def blog_block_md5() -> set[str]:
    digests = set()
    for article in BLOG.glob("*.md"):
        for _, body in BLOCK_RE.findall(article.read_text(encoding="utf-8")):
            digests.add(hashlib.md5((body.rstrip("\n") + "\n").encode()).hexdigest())
    return digests


def article_dirs():
    return sorted(p for p in REPO.glob("[0-9][0-9]-*/*") if p.is_dir())


@unittest.skipUnless(BLOG.is_dir(), "缺少博文目录，跳过溯源测试")
class TestProvenance(unittest.TestCase):
    def test_every_file_traces_back_to_blog(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"],
            cwd=REPO, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("篇目代码全部可追溯到博文代码块", result.stdout)

    def test_provenance_entries_are_well_formed(self):
        data = json.loads((REPO / "docs" / "maintaining" / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(data["version"], 2)
        for rel, entry in data["files"].items():
            with self.subTest(file=rel):
                self.assertTrue((REPO / rel).is_file(), "登记了但文件不存在")
                self.assertTrue((BLOG / entry["article"]).is_file(), "博文不存在")
                self.assertIsInstance(entry["blocks"], list)
                self.assertGreaterEqual(len(entry["blocks"]), 2, "单块不需要拼接登记")
                self.assertTrue(entry.get("why"), "缺少拼接理由")

    def test_unregistered_files_still_match_a_block(self):
        """未登记的文件要么逐字命中博文块，要么必须在 PROVENANCE.json 里。"""
        registered = set(
            json.loads((REPO / "docs" / "maintaining" / "provenance.json").read_text(encoding="utf-8"))["files"]
        )
        known = blog_block_md5()
        for path in article_dirs():
            for file in path.iterdir():
                if file.suffix not in {".py", ".cu", ".cuh", ".cpp", ".c"}:
                    continue
                rel = file.relative_to(REPO).as_posix()
                with self.subTest(file=rel):
                    if rel in registered:
                        continue
                    digest = hashlib.md5(file.read_bytes()).hexdigest()
                    self.assertIn(digest, known, f"{rel} 既未登记，也不匹配任何博文块")

    def test_standalone_entries_are_well_formed(self):
        """独立项目按自己的构建与测试验收，登记必须写清构建入口，且入口真实存在。

        这条替代了旧的「pending（待博文）」：完整工程不以是否发表过文章作为合格标准，
        但它必须能独立构建——否则「独立项目」四个字就是空话。"""
        data = json.loads((REPO / "docs" / "maintaining" / "provenance.json").read_text(encoding="utf-8"))
        entries = data.get("standalone", [])
        self.assertTrue(entries, "standalone 段不应为空；确有独立项目时请登记")
        for entry in entries:
            with self.subTest(paths=entry.get("paths")):
                self.assertTrue(entry.get("why", "").strip(), "缺少「为什么它是独立项目」")
                prefixes = entry.get("paths")
                self.assertIsInstance(prefixes, list)
                self.assertGreaterEqual(len(prefixes), 1)
                for prefix in prefixes:
                    self.assertTrue(prefix.endswith("/"), "前缀必须以 / 结尾，否则会误伤同名前缀")
                    hits = [p for p in REPO.glob(f"{prefix}**/*") if p.is_file()]
                    self.assertTrue(hits, f"{prefix} 下没有任何文件，登记已失效")
                built = entry.get("built_by", "")
                self.assertTrue(built, "缺少 built_by 构建入口")
                self.assertTrue((REPO / built).is_file(), f"构建入口不存在：{built}")


if __name__ == "__main__":
    unittest.main()
