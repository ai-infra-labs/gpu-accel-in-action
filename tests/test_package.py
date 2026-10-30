"""cuda_kernels 包的契约测试：只用标准库，不需要 GPU。

守的是三条不变量：

1. 目录扫描结果 = 章节目录的真实内容（不维护第二份清单）；
2. 模块名唯一（ops 的属性名靠它，重名会让 `ops.x` 指向不确定的文件）；
3. load() 不会在 import 时把演示代码跑起来（只接受带 __main__ 守卫的文件）。
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import cuda_kernels as ck  # noqa: E402  （需要先加 sys.path）


class TestCatalog(unittest.TestCase):
    def test_articles_match_disk(self):
        on_disk = sorted(p for p in (REPO / "examples").glob("0[1-9]-*/[0-9][0-9][0-9]-*") if p.is_dir())
        self.assertEqual([a.dir for a in ck.articles()], on_disk)

    def test_ids_are_unique_and_sorted(self):
        ids = [a.id for a in ck.articles()]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_module_names_are_unique(self):
        """同名文件会让 ops.<name> 指向不确定的篇目，必须在入库时就挡住。"""
        names = [p.stem for a in ck.articles() for p in a.files]
        dup = {n for n in names if names.count(n) > 1}
        self.assertEqual(dup, set(), f"模块名重复：{sorted(dup)}")

    def test_find_by_id_slug_and_partial(self):
        self.assertEqual(ck.locate(11).name, "011-tiled-matmul-shared-memory")
        self.assertEqual(ck.locate("011").name, "011-tiled-matmul-shared-memory")
        self.assertEqual(ck.locate("tiled-matmul-shared-memory").name,
                         "011-tiled-matmul-shared-memory")
        # 部分匹配只在唯一时生效
        self.assertEqual(ck.locate("coalescing").name, "061-coalescing-lab")

    def test_find_rejects_unknown_and_ambiguous(self):
        with self.assertRaises(KeyError):
            ck.locate(999)
        with self.assertRaises(KeyError):
            ck.locate("compression")  # 命中多篇，不猜


class TestRuntime(unittest.TestCase):
    def test_load_rejects_cuda_source(self):
        cu = ck.locate(56) / "affine_kernel.cu"
        if not cu.is_file():
            self.skipTest("056 的 .cu 不在库里")
        with self.assertRaises(ValueError):
            ck.load("affine_kernel")

    def test_load_refuses_module_without_main_guard(self):
        with self.assertRaises(RuntimeError):
            ck.load("tiled_matmul")

    def test_load_works_for_stdlib_module(self):
        module = ck.load("gpu_task_dispatch")
        self.assertTrue(callable(module.main), "044 的主入口应可调用")
        self.assertIn("cuda_kernels", module.__name__)

    def test_ops_namespace_matches_py_files(self):
        from cuda_kernels import ops

        expected = sorted(p.stem for a in ck.articles() for p in a.py_files)
        self.assertEqual(ops.names(), expected)

    def test_ops_missing_attribute_explains_cuda(self):
        from cuda_kernels import ops

        with self.assertRaises(AttributeError) as ctx:
            ops.affine_kernel
        self.assertIn("nvcc", str(ctx.exception))


class TestCli(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "cuda_kernels", *args],
                              cwd=REPO, capture_output=True, text=True)

    def test_list(self):
        out = self.run_cli("list")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("011-tiled-matmul-shared-memory", out.stdout)

    def test_locate_prints_path(self):
        out = self.run_cli("locate", "011")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertTrue(Path(out.stdout.strip()).is_dir())

    def test_unknown_key_exits_nonzero(self):
        out = self.run_cli("locate", "999")
        self.assertEqual(out.returncode, 2)
        self.assertIn("错误", out.stderr)


if __name__ == "__main__":
    unittest.main()
