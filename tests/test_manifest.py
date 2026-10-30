"""清单与检索的校验。

覆盖：articles.json 的完整性、与 blog/ 和仓库目录的双向一致性、索引/清单的
--check 稳定性，以及 search.py 的按标签检索。零三方依赖。
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import corpus  # noqa: E402
import search  # noqa: E402

MANIFEST = REPO / "docs" / "_data" / "articles.json"

REQUIRED_KEYS = (
    "id", "slug", "title", "chapter", "chapter_name", "chapter_dir",
    "blog_path", "repo_dir", "ingest_status", "verify_status",
    "mechanisms", "deps", "block_count", "max_block", "code_files",
)

INGEST_ALLOWED = {
    "已入库", "已入库（已实测）", "已入库（已实测·合成数据）",
    "已入库（部分实测）", "已入库（无独立代码）", "待拼装", "待评估",
}


class TestManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.arts = cls.data["articles"]

    def test_manifest_exists_and_count(self):
        self.assertEqual(len(self.arts), 81)
        for a in self.arts:
            for k in REQUIRED_KEYS:
                self.assertIn(k, a, f"缺字段 {k}: {a.get('id')}")
            self.assertIsInstance(a["id"], int)
            self.assertIsInstance(a["block_count"], int)
            self.assertIsInstance(a["max_block"], int)
            self.assertGreaterEqual(a["block_count"], 0)
            self.assertGreaterEqual(a["max_block"], 0)

    def test_ingest_status_values(self):
        for a in self.arts:
            self.assertIn(a["ingest_status"], INGEST_ALLOWED, a["id"])

    def test_covers_blog(self):
        blog_files = {p.name for p in (REPO / "blog").glob("*.md")
                      if not p.name.startswith("_")}
        covered = set()
        for a in self.arts:
            path = REPO / a["blog_path"]
            self.assertTrue(path.is_file(), f"blog 路径不存在：{a['blog_path']}")
            covered.add(Path(a["blog_path"]).name)
        self.assertEqual(covered, blog_files, "清单与 blog/ 文件不一致")

    def test_repo_dirs_consistent(self):
        actual = {str(p.relative_to(REPO)) for p in (REPO / "examples").glob("*/[0-9][0-9][0-9]-*")
                  if p.is_dir()}
        listed = {a["repo_dir"] for a in self.arts if a["repo_dir"]}
        self.assertEqual(listed, actual, "repo_dir 与仓库实际目录不一致")
        self.assertEqual(len(listed), 53, "入库目录数应为 53")
        for d in listed:
            self.assertTrue((REPO / d).is_dir())
        # 没有 repo_dir 的应是 28 篇待拼装/待评估
        self.assertEqual(len(self.arts) - len(listed), 28)

    def test_index_check(self):
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "gen_index.py"), "--check"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_manifest_check(self):
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "gen_manifest.py"), "--check"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_atlas_check(self):
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "gen_atlas.py"), "--check"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_search_by_tag(self):
        ns = __import__("types").SimpleNamespace(
            id=None, chapter=None, status=None, tag="nccl", query=None, list=False)
        hits = [a for a in self.arts if search.matches(a, ns)]
        self.assertEqual(sorted(a["id"] for a in hits), [47, 48])

    def test_search_free_text(self):
        ns = __import__("types").SimpleNamespace(
            id=None, chapter=None, status=None, tag=None, query="nvl", list=False)
        hits = [a for a in self.arts if search.matches(a, ns)]
        self.assertEqual([a["id"] for a in hits], [66])


if __name__ == "__main__":
    unittest.main()
