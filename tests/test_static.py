"""静态检查：语法 + 依赖分层 + 硬件需求，全部在无 GPU 环境可跑。

`static_check.py` 是仓库自己的工具，这里只把它当契约用：
它的退出码为 0、且不产生警告，才说明新加的代码没有引入未登记的依赖。
"""

from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "static_check.py"


class TestStaticCheck(unittest.TestCase):
    def test_static_check_passes_without_warnings(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=REPO, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("警告", result.stdout, "存在未登记的依赖，请先在 static_check.py 与 requirements/ 里登记")
        self.assertIn("语法检查通过", result.stdout)

    def test_all_python_files_parse(self):
        offenders = []
        for path in REPO.rglob("*.py"):
            if ".git" in path.parts:
                continue
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                offenders.append(f"{path.relative_to(REPO)}:{exc.lineno} {exc.msg}")
        self.assertEqual(offenders, [], "存在语法错误")

    def test_no_replacement_characters(self):
        """中文注释写坏会变成 U+FFFD，肉眼在长文件里很难发现。"""
        offenders = []
        for pattern in ("*.py", "*.md", "*.txt"):
            for path in REPO.rglob(pattern):
                if ".git" in path.parts:
                    continue
                if "\ufffd" in path.read_text(encoding="utf-8", errors="replace"):
                    offenders.append(str(path.relative_to(REPO)))
        self.assertEqual(offenders, [], "文件包含 U+FFFD 乱码")


if __name__ == "__main__":
    unittest.main()
