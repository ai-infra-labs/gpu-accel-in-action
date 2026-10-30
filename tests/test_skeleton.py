"""骨架一致性：四个可机械判定的要素，每个文件都要有。

为什么只查四个而不是五个：「参数区」的口径太宽（模块级常量、`argparse`、
`main()` 里的局部常量都算），本测试用 AST 判定，规则写在 `has_params`。
「校验」与「入口」的例外是真实存在的，逐个登记理由，防止悄悄放宽。
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CODE_SUFFIXES = {".py"}

# 允许没有显式 assert 的文件：它们把校验结果打印出来（[验证] 行 / ok 布尔）
NO_CHECK = {
    "examples/01-gpu-programming/003-cpu-gpu-cooperative/cpu_gpu_cooperative.py": "校验以 [验证] 打印的等式呈现，失败由人眼比对",
    "examples/01-gpu-programming/005-image-hflip/image_hflip.py": "校验以 `结果正确: True/False` 打印，无 assert",
}

# 允许没有 print 的文件：纯模块，被别的脚本 import
NO_PRINT = {
    "examples/06-custom-operators/056-custom-op-vs-eager/affine.py": "扩展加载模块，只做 load 并把算子挂到 torch.ops",
}

# 允许没有 __main__ 的文件，理由见 docs/maintaining/example-conventions.md
NO_MAIN = {
    "examples/01-gpu-programming/001-fc-forward-backward/fc_forward_backward.py": "博文附录为模块级脚本",
    "examples/01-gpu-programming/002-fc-backprop-sgd/fc_backprop_sgd.py": "博文附录为模块级脚本",
    "examples/01-gpu-programming/005-image-hflip/image_hflip.py": "博文附录为模块级脚本",
    "examples/01-gpu-programming/006-image-crop-roi/image_crop_roi.py": "博文附录为模块级脚本",
    "examples/01-gpu-programming/007-conv2d-correlation/conv2d_correlation.py": "博文附录为模块级脚本",
    "examples/01-gpu-programming/009-fused-preprocess/fused_preprocess.py": "博文附录为模块级脚本",
    "examples/01-gpu-programming/011-tiled-matmul-shared-memory/tiled_matmul.py": "博文附录为模块级脚本",
    "examples/05-parallel-distributed/051-parallel-group-check/parallel_groups_check.py": "断言与打印都在模块级，本身是检查脚本",
    "examples/06-custom-operators/056-custom-op-vs-eager/affine.py": "被 import 的模块",
    "examples/09-medical-imaging/079-ct-amp-check/fp16_numpy_check.py": "纯格式检查，无可调参数",
}


def code_files() -> list[Path]:
    return sorted(
        p for p in (REPO / "examples").glob("[0-9][0-9]-*/*/*")
        if p.is_file() and p.suffix in CODE_SUFFIXES
    )


def has_params(tree: ast.Module, source: str) -> bool:
    """参数区：模块级常量、argparse，或函数/类里的常量与字段。"""
    if "argparse" in source:
        return True
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.ClassDef)):
            return True
        if isinstance(node, ast.FunctionDef):
            if any(isinstance(sub, (ast.Assign, ast.AnnAssign)) for sub in node.body):
                return True
    return False


class TestSkeleton(unittest.TestCase):
    def test_parameters_section_exists(self):
        offenders = [str(f.relative_to(REPO)) for f in code_files()
                     if not has_params(ast.parse(f.read_text(encoding="utf-8")),
                                       f.read_text(encoding="utf-8"))]
        self.assertEqual(offenders, [], "文件里找不到参数区（常量或 argparse）")

    def test_every_file_verifies_itself(self):
        offenders = []
        for path in code_files():
            rel = path.relative_to(REPO).as_posix()
            source = path.read_text(encoding="utf-8")
            if rel in NO_CHECK:
                continue
            if not any(k in source for k in ("assert ", "np.testing", "torch.testing", "raise ")):
                offenders.append(rel)
        self.assertEqual(offenders, [], "文件既没有 assert/np.testing，也没有登记到 NO_CHECK")

    def test_every_file_reports(self):
        offenders = [p.relative_to(REPO).as_posix() for p in code_files()
                     if "print(" not in p.read_text(encoding="utf-8")
                     and p.relative_to(REPO).as_posix() not in NO_PRINT]
        self.assertEqual(offenders, [], "文件没有任何输出，也没有登记到 NO_PRINT")

    def test_entry_point_or_whitelist(self):
        offenders = []
        for path in code_files():
            rel = path.relative_to(REPO).as_posix()
            if rel in NO_MAIN:
                continue
            if "__main__" not in path.read_text(encoding="utf-8"):
                offenders.append(rel)
        self.assertEqual(offenders, [], "缺 __main__ 且未登记到 NO_MAIN")

    def test_whitelists_are_not_stale(self):
        """白名单里的文件必须仍然存在，防止改了名之后白名单悄悄失效。"""
        existing = {p.relative_to(REPO).as_posix() for p in code_files()}
        for name, table in (("NO_CHECK", NO_CHECK), ("NO_PRINT", NO_PRINT), ("NO_MAIN", NO_MAIN)):
            stale = sorted(set(table) - existing)
            self.assertEqual(stale, [], f"{name} 里有已不存在的文件：{stale}")
            for rel, why in table.items():
                self.assertTrue(why.strip(), f"{name} 的 {rel} 缺少理由")


if __name__ == "__main__":
    unittest.main()
