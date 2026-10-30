"""懒加载的算子/脚本命名空间。

    from cuda_kernels import ops
    ops.gpu_task_dispatch.main()        # 属性访问时才 import，缺依赖也不会拖垮整包

属性名 = 文件名去后缀（仓库内不重名，`tests/test_package.py` 会守这条不变量）。
只有 .py 能这样取；`.cu` 见 `cuda_kernels.runtime` 的说明——它要交给 nvcc，不能 import。

想先看有什么：

    ops.names()      # 全部可加载的模块名
    ops.describe()   # 打印每个模块属于哪一篇
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from .. import _catalog, runtime

_PY: dict[str, Path] = {
    p.stem: p for a in _catalog.articles() for p in a.py_files
}


def names() -> list[str]:
    """全部可加载的模块名（按名字排序）。"""
    return sorted(_PY)


def describe() -> None:
    """打印模块 → 篇目 的对应关系。"""
    owner = {p.stem: (a.id, a.slug) for a in _catalog.articles() for p in a.py_files}
    for name in names():
        num, slug = owner[name]
        print(f"{name:32s} {num:03d}  {slug}")


def __getattr__(name: str) -> ModuleType:
    if name in _PY:
        module = runtime.load(_PY[name])
        globals()[name] = module  # 缓存，后续访问不再 import
        return module
    hint = ""
    cu = {p.stem for a in _catalog.articles() for p in a.cu_files}
    if name in cu:
        hint = "（这是 CUDA 源码，不能 import；按 docs/getting-started.md 用 nvcc / CMake 构建）"
    raise AttributeError(f"ops 里没有 {name!r}{hint}；ops.names() 看全部")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_PY))
