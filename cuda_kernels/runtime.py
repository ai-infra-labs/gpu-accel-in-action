"""加载与运行：把章节目录里的脚本当成模块用，但不复制一行代码。

为什么是「按路径加载」而不是把源码搬进包里：

    仓库的硬规矩是「代码与博文逐字一致」，靠 md5 对账（scripts/build_from_blocks.py）。
    一旦把 .py 复制一份进 cuda_kernels/，就会出现两个会各自演化的副本，
    md5 契约立刻失效。所以这里只做「视图」——源文件永远只有章节目录那一份。

代价要讲清楚：脚本不是库。多数文件在模块级就写死了演示规模与 print，
import 它们等于跑一遍。因此 load() 默认只接受带 `if __name__ == "__main__"` 守卫的文件，
没有守卫的会明确拒绝，而不是让读者在 import 时莫名跑起 benchmark。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

GUARD_RE = "if __name__"


def has_main_guard(path: Path) -> bool:
    """文件是否有 __main__ 守卫（有守卫才可能在 import 时无副作用）。"""
    if path.suffix != ".py":
        return False
    return GUARD_RE in path.read_text(encoding="utf-8")


def load(path: Path, *, allow_side_effects: bool = False) -> ModuleType:
    """按文件路径把脚本载入为模块。

    allow_side_effects=False（默认）时，没有 __main__ 守卫的文件会被拒绝——
    这类文件的模块级代码就是演示本身，import 会直接跑 benchmark 或申请显存。
    """
    path = Path(path)
    if path.suffix != ".py":
        raise ValueError(
            f"{path.name} 是 CUDA 源码，不能 import；用 run() 交给 nvcc，"
            "或按 docs/getting-started.md 走 CMake"
        )
    if not allow_side_effects and not has_main_guard(path):
        raise RuntimeError(
            f"{path.name} 没有 `if __name__ == \"__main__\"` 守卫，import 会直接执行演示代码。"
            "确需加载请传 allow_side_effects=True，或改用 run()。"
        )

    name = f"cuda_kernels._loaded.{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {path} 建立模块规格")
    module = importlib.util.module_from_spec(spec)

    # 同目录脚本之间会互相 import（如 070 复用 069 的封装），
    # 把所在目录临时放进 sys.path，导入完成后还原。
    parent = str(path.parent)
    inserted = parent not in sys.path
    if inserted:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass
    return module


def run(path: Path, args: list[str] | None = None, *, timeout: float | None = None) -> subprocess.CompletedProcess:
    """在源文件所在目录用当前解释器运行它（cwd 很关键：脚本按相对路径读数据）。"""
    path = Path(path)
    if path.suffix == ".py":
        cmd = [sys.executable, path.name, *(args or [])]
    else:
        raise ValueError(
            f"{path.name} 不是 Python 脚本；CUDA 源码请用 nvcc 或 CMake 构建后运行，见 docs/getting-started.md"
        )
    return subprocess.run(cmd, cwd=path.parent, text=True, timeout=timeout)
