"""cuda_kernels —— 《大模型训练与推理加速实战》配套代码的加载入口。

它**不是**算子库：仓库里的源码仍然只有章节目录下那一份（与博文逐字一致），
本包提供的是「按名字找到它、加载它、运行它」的能力。

    import cuda_kernels as ck

    ck.locate("011")                    # 第 11 篇的代码目录（Path）
    ck.files("011")                     # 目录里的代码文件
    ck.locate("coalescing")             # 也可以按 slug 找
    mod = ck.load("gpu_task_dispatch")  # 载入为模块（要求有 __main__ 守卫）
    mod.main()

    ck.run("gpu_task_dispatch")         # 等价于在该目录下 python gpu_task_dispatch.py

    from cuda_kernels import ops
    ops.tiled_matmul                    # 懒加载：取属性时才 import

为什么不做成可 `pip install` 的算子库：56/062/063 这些 kernel 是**教学脚本**，
模块级就带演示规模与 print；包成库会逼着改源码，而源码一旦改动就要同步博文，
md5 契约（scripts/build_from_blocks.py）就断了。取舍与替代方案见 docs/reference/package.md。
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from . import _catalog, runtime

__version__ = "1.1.0"

__all__ = [
    "Article",
    "articles",
    "locate",
    "files",
    "load",
    "run",
    "repo_root",
    "runtime",
]


def repo_root() -> Path:
    """仓库根目录（本包位于仓库根下，因此是包的上一级）。"""
    return _catalog.REPO_ROOT


def articles() -> tuple[_catalog.Article, ...]:
    """全部有代码目录的篇目，按篇号升序。"""
    return _catalog.articles()


def locate(key: str | int) -> Path:
    """按篇号 / slug 查代码目录。"""
    return _catalog.find(key).dir


def files(key: str | int) -> tuple[Path, ...]:
    """按篇号 / slug 查目录内的代码文件（.py / .cu）。"""
    return _catalog.find(key).files


def load(name: str, *, allow_side_effects: bool = False) -> ModuleType:
    """按模块名（文件名去后缀）把脚本载入为模块。"""
    return runtime.load(_catalog.find_module(name), allow_side_effects=allow_side_effects)


def run(name: str, args: list[str] | None = None) -> int:
    """在脚本所在目录运行它，返回退出码。"""
    path = _catalog.find_module(name)
    return runtime.run(path, args).returncode
