"""篇目目录索引：把「章节目录里有什么」变成可查询的数据。

刻意不维护第二份清单——所有信息都从目录名和文件名推导：

    01-gpu-programming/011-tiled-matmul-shared-memory/tiled_matmul.py
    └─ chapter=1  └─ id=11  └─ slug=tiled-matmul-shared-memory  └─ module=tiled_matmul

篇号和 slug 的单一事实来源仍是目录名，所以本模块永远不会和 `docs/reference/article-index.md` 打架。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_SUFFIXES = {".py", ".cu"}
ARTICLE_RE = re.compile(r"^(\d{3})-(.+)$")


@dataclass(frozen=True)
class Article:
    """一篇博文对应的代码目录。"""

    id: int
    slug: str
    chapter: int
    chapter_dir: str
    dir: Path
    files: tuple[Path, ...] = field(default=())

    @property
    def py_files(self) -> tuple[Path, ...]:
        return tuple(p for p in self.files if p.suffix == ".py")

    @property
    def cu_files(self) -> tuple[Path, ...]:
        return tuple(p for p in self.files if p.suffix == ".cu")


def _scan() -> tuple[Article, ...]:
    out: list[Article] = []
    for d in sorted((REPO_ROOT / "examples").glob("0[1-9]-*/[0-9][0-9][0-9]-*")):
        if not d.is_dir():
            continue
        m = ARTICLE_RE.match(d.name)
        if not m:
            continue
        files = tuple(sorted(p for p in d.iterdir() if p.suffix in CODE_SUFFIXES))
        out.append(Article(
            id=int(m.group(1)),
            slug=m.group(2),
            chapter=int(d.parent.name[:2]),
            chapter_dir=d.parent.name,
            dir=d,
            files=files,
        ))
    return tuple(out)


_CACHE: tuple[Article, ...] | None = None


def articles() -> tuple[Article, ...]:
    """全部篇目（按篇号升序），结果缓存在模块内。"""
    global _CACHE
    if _CACHE is None:
        _CACHE = _scan()
    return _CACHE


def _norm(key: str) -> str:
    return key.strip().lower().replace("_", "-")


def find(key: str | int) -> Article:
    """按篇号（11 / "011"）、slug 或目录名查一篇；找不到抛 KeyError 并列出候选。"""
    if isinstance(key, int) or str(key).isdigit():
        num = int(key)
        for a in articles():
            if a.id == num:
                return a
        raise KeyError(f"没有第 {num:03d} 篇的代码目录（片段型与纯文字篇不建目录）")

    want = _norm(str(key))
    for a in articles():
        if _norm(a.slug) == want or _norm(a.dir.name) == want:
            return a
    # 允许用 slug 的一部分查（"coalescing" → coalescing-lab），但只在唯一匹配时生效：
    # 猜错篇目比报错更糟，所以歧义时仍然抛错并列出候选。
    hits = [a for a in articles() if want in _norm(a.slug)]
    if len(hits) == 1:
        return hits[0]
    hint = f"；相近的有：{', '.join(a.slug for a in hits[:5])}" if hits else ""
    raise KeyError(f"找不到篇目 {key!r}{hint}")


def modules() -> dict[str, Path]:
    """模块名 → 源文件。模块名即文件名去后缀，仓库内不重名。"""
    out: dict[str, Path] = {}
    for a in articles():
        for p in a.files:
            out[p.stem] = p
    return out


def find_module(name: str) -> Path:
    """按模块名（不带后缀）找源文件。"""
    mods = modules()
    if name in mods:
        return mods[name]
    want = name.replace("-", "_")
    if want in mods:
        return mods[want]
    near = [k for k in mods if want in k][:5]
    hint = f"；相近的有：{', '.join(near)}" if near else ""
    raise KeyError(f"找不到模块 {name!r}{hint}")
