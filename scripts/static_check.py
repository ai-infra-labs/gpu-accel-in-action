#!/usr/bin/env python3
"""全仓静态检查：语法 + 依赖分层 + 硬件需求。

本机没有 NVIDIA GPU 也能跑，用于在无卡环境下拦截最基础的三类错误：
  1. 语法错误（ast.parse 失败）
  2. 依赖越层（脚本引用了未在本层 requirements 声明的包）
  3. 硬件需求与依赖层不一致（装了 CPU 依赖却调用 .cuda()）

它不做数值正确性验证。凡是需要 GPU 的部分，必须到 N 卡机器上跑，
结果记入各篇的 expected_output.md。
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 依赖分层：与 requirements/ 下的文件一一对应。
# 每个 requirements 文件都是自足的（含 numpy/scipy 这类公共地基），
# 所以文件只需归到"最高的那一层"，不必同时引用多个文件。
TIER = {
    "advanced": {"tensorrt", "triton", "numba", "mpi4py", "monai", "modelopt"},
    "gpu": {"pycuda", "cupy", "cupyx"},
    "cpu": {"torch", "torchvision", "onnx", "onnxruntime", "onnxscript", "matplotlib", "nibabel"},
}

# 多层共用、不足以判定归属的包。
SHARED = {"numpy", "scipy"}

# 判定优先级：依赖越"重"的层优先。numba 脚本同时用 scipy，应归 advanced。
TIER_ORDER = ("advanced", "gpu", "cpu")

# 硬件需求与依赖层是两件事：torch 属于 CPU 层，但调了 .cuda() 就必须有 N 卡。
# 这里只做保守的文本特征匹配，宁可漏报也不误报成"确定需要 GPU"。
GPU_PATTERNS = (
    r"import pycuda",
    r"\.cuda\(\s*\)",
    r"\.cuda\(\s*\d",
    r"torch\.device\(\s*[\"']cuda",
    r"[\"']cuda:\d",
    r"@cuda\.jit",
    r"cuda\.to_device",
    r"cuda\.device_array",
    r"import tensorrt",
    r"numba\.cuda",
    r"import cupy",
    r"cupy\.cuda",
    r"torch\.cuda\.(synchronize|set_device|Stream|Event)",
    r"device\s*=\s*[\"']cuda",
    r"\.to\(\s*[\"']cuda",
)

# 仓库内的本地模块，不是第三方依赖：
#   reduction_scan —— 010 的验证脚本 import 同目录的主脚本
#   affine         —— 056 的基准脚本 import 同目录的扩展加载模块
#   run_engine     —— 070 复用 069 的引擎封装（用 PYTHONPATH 指过去）
#   static_check   —— scripts/gen_atlas.py 复用同一套分层规则
#   corpus         —— scripts 下的语料中央模块，被 gen_atlas/gen_index/
#                     gen_manifest/search 复用
#   search         —— scripts 下的检索工具，被 test_manifest.py 复用
#   gen_verify     —— scripts/gen_chapters.py 复用同一套实测状态归类
#   cuda_kernels   —— 仓库自带的加载包（不入篇目代码，见 docs/reference/package.md）
LOCAL_MODULES = {"reduction_scan", "affine", "run_engine", "static_check", "corpus", "search",
                 "gen_verify", "cuda_kernels"}

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "build", "onnx_demo"}


def iter_py_files() -> list[Path]:
    files = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def top_level_imports(tree: ast.AST) -> set[str]:
    """只取模块根名：from pycuda.driver import x → pycuda。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def classify(names: set[str]) -> tuple[str, list[str]]:
    """返回该文件所需的环境层与未登记依赖。"""
    unknown: list[str] = []
    found: set[str] = set()
    for name in names:
        if name in sys.stdlib_module_names or name in SHARED or name in LOCAL_MODULES:
            continue
        for tier in TIER_ORDER:
            if name in TIER[tier]:
                found.add(tier)
                break
        else:
            unknown.append(name)

    tier = next((t for t in TIER_ORDER if t in found), "stdlib")
    return tier, sorted(unknown)


def needs_gpu(source: str) -> bool:
    """保守判断脚本是否调用 CUDA API，与它依赖哪一层无关。"""
    return any(re.search(p, source) for p in GPU_PATTERNS)


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[tuple[str, int, str, str]] = []

    for path in iter_py_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            errors.append(f"{rel}:{exc.lineno}: 语法错误 -> {exc.msg}")
            continue
        except UnicodeDecodeError:
            errors.append(f"{rel}: 不是 UTF-8 编码")
            continue

        tier, unknown = classify(top_level_imports(tree))
        for name in unknown:
            warnings.append(f"{rel}: 未登记的依赖 {name!r}")

        # scripts/ 与 tests/ 是本仓库的工具，不是文章代码，不参与硬件判定
        is_tool = rel.startswith(("scripts/", "tests/"))
        hw = "GPU" if (not is_tool and needs_gpu(source)) else "CPU"
        # 标星：依赖落在 CPU 层却调用 CUDA API —— 装依赖不够，还得有卡
        if hw == "GPU" and tier == "cpu":
            hw = "GPU*"

        rows.append((rel, len(source.splitlines()), tier, hw))

    print(f"检查 {len(rows)} 个 Python 文件\n")
    width = max((len(r[0]) for r in rows), default=10)
    for rel, lines, tier, hw in rows:
        print(f"  {rel:<{width}}  {lines:>4} 行  [{tier:>8s}]  需 {hw}")
    if any(r[3] == "GPU*" for r in rows):
        print("\n  * 依赖属 CPU 层，但调用了 CUDA API：装 requirements 之后仍需 N 卡")

    if warnings:
        print(f"\n警告 {len(warnings)} 条：")
        for w in warnings:
            print("  ! " + w)

    if errors:
        print(f"\n失败 {len(errors)} 条：")
        for e in errors:
            print("  x " + e)
        return 1

    print("\n语法检查通过。数值正确性仍需在 GPU 环境补测。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
