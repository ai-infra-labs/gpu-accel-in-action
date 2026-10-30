#!/usr/bin/env python3
"""仓库语料的单一数据源：扫描 blog/ 与九章目录，产出每篇的结构化字段。

设计目标（对应「正规 / 可读 / 检索快」）：
- 单一的标签词表（MECHANISMS / LEGEND / CHAPTERS）只在此定义，gen_atlas、
  gen_index、search 都从这里导入，避免词表漂移。
- collect() 不读入任何手工大表，全部由仓库自身推导；只有「入库状态」这类
  策展字段由 articles.json 提供（见 gen_manifest.py 的引导逻辑）。

用法（被其它脚本 import，不直接运行）：
    from corpus import collect, MECHANISMS, LEGEND, CHAPTERS
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BLOG_DIR = REPO_ROOT / "blog"
ARTICLE_INDEX = REPO_ROOT / "docs" / "reference" / "article-index.md"

CODE_SUFFIXES = {".py", ".cu", ".c", ".cuh", ".cpp"}
CODE_LANGS = {"python", "cpp", "cuda", "c", "cuh"}
BLOCK_RE = re.compile(r"```([A-Za-z0-9_+\-]*)[ \t]*\n(.*?)```", re.S)

# 章节归属：与 docs/learning-path.md 的九章一致
CHAPTERS = [
    (1, "GPU 编程基础", "01-gpu-programming", 1, 19),
    (2, "模型压缩", "02-model-compression", 20, 26),
    (3, "推理服务与部署", "03-inference-serving", 27, 35),
    (4, "端侧推理", "04-edge-inference", 36, 41),
    (5, "并行与分布式", "05-parallel-distributed", 42, 51),
    (6, "自定义算子", "06-custom-operators", 52, 59),
    (7, "访存与传输", "07-memory-and-transfer", 60, 66),
    (8, "TensorRT 部署", "08-tensorrt-deploy", 67, 73),
    (9, "医疗影像实战", "09-medical-imaging", 74, 81),
]

# 机制标签：一篇 1–3 个，用于「按能力反查」。词表要克制，新增标签必须同时补 LEGEND。
MECHANISMS = {
    1: ["thread-mapping"], 2: ["thread-mapping", "coalescing"], 3: ["async-copy", "latency-breakdown"],
    4: ["latency-breakdown", "transfer"], 5: ["thread-mapping"], 6: ["thread-mapping"], 7: ["thread-mapping"],
    8: ["thread-mapping", "memory-reuse"], 9: ["fusion", "memory-reuse"], 10: ["reduction", "sync"],
    11: ["shared-memory", "sync"], 13: ["transfer", "latency-breakdown"], 14: ["launch-overhead", "sync"],
    15: ["shared-memory"], 17: ["data-race", "thread-mapping"], 18: ["sync", "stability"],
    19: ["sync", "boundary"], 21: ["weight-quant", "dsl-numba"], 22: ["distillation"],
    23: ["params-vs-flops"], 26: ["csr-sparse"], 28: ["tensorrt-build", "latency-breakdown"],
    29: ["tensorrt-build", "latency-breakdown"], 42: ["async-copy"], 43: ["memory-accounting", "occupancy"],
    44: ["scheduling"], 47: ["nccl"], 48: ["tp-parallel", "nccl"], 51: ["distributed-3d"],
    56: ["fusion", "dsl-torch", "launch-overhead"], 60: ["memory-accounting"], 61: ["coalescing"],
    62: ["shared-memory", "occupancy"], 63: ["constant-memory", "coalescing"],
    64: ["pinned-memory", "transfer"], 65: ["async-copy"], 66: ["p2p-copy", "transfer"],
    67: ["onnx-export"], 68: ["tensorrt-build", "fusion"], 69: ["tensorrt-runtime"],
    70: ["latency-breakdown", "tensorrt-runtime"], 71: ["int8-quant"], 72: ["pruning", "transfer"],
    73: ["pruning", "params-vs-flops"], 74: ["nifti-geometry"], 75: ["inference-pipeline", "dice-metric"],
    76: ["fusion", "transfer", "dsl-cupy"], 77: ["sliding-window"], 78: ["training-step", "dice-metric"],
    79: ["amp"], 80: ["async-copy", "pinned-memory"], 81: ["dice-metric"],
}
LEGEND = {
    "thread-mapping": "线程与数据的映射", "shared-memory": "共享内存分块", "reduction": "归约/前缀和",
    "sync": "同步与双缓冲", "data-race": "数据竞争", "coalescing": "合并访存", "constant-memory": "常量内存广播",
    "fusion": "算子融合", "memory-reuse": "中间结果复用", "launch-overhead": "启动开销", "transfer": "主机↔设备传输",
    "pinned-memory": "页锁定内存", "async-copy": "异步与重叠", "p2p-copy": "点对点拷贝", "memory-accounting": "显存记账",
    "occupancy": "占用率", "stability": "数值稳定性", "boundary": "边界条件", "tensorrt-build": "引擎构建",
    "tensorrt-runtime": "运行时与形状", "onnx-export": "导出一致性", "int8-quant": "INT8 量化", "weight-quant": "权重存储量化",
    "pruning": "剪枝与稀疏", "csr-sparse": "CSR 稀疏格式", "params-vs-flops": "参数量 vs 计算量", "nccl": "集合通信",
    "tp-parallel": "张量并行", "distributed-3d": "3D 并行", "scheduling": "请求调度", "latency-breakdown": "时延拆分",
    "amp": "混合精度（AMP）", "sliding-window": "滑窗推理", "nifti-geometry": "医学影像几何", "dice-metric": "分割评估",
    "inference-pipeline": "推理链路", "training-step": "训练一步", "distillation": "知识蒸馏",
    "dsl-numba": "Numba 抽象", "dsl-cupy": "CuPy 抽象", "dsl-torch": "PyTorch 抽象",
}


def chapter_of(num: int) -> tuple[int, str, str]:
    for idx, name, slug, lo, hi in CHAPTERS:
        if lo <= num <= hi:
            return idx, name, slug
    return 0, "未分章", ""


def iter_blog_articles() -> list[tuple[int, str, Path]]:
    """返回 [(篇号, slug, 博文路径)]，按篇号升序。"""
    out = []
    for p in BLOG_DIR.glob("*.md"):
        if p.name.startswith("_"):
            continue
        m = re.match(r"^(\d{3})-(.+)\.md$", p.name)
        if not m:
            continue
        out.append((int(m.group(1)), m.group(2), p))
    return sorted(out, key=lambda t: t[0])


def article_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def block_stats(path: Path) -> tuple[int, int]:
    """代码块数量与最长代码块行数（仅统计 python/cpp/cuda/c/cuh）。"""
    blocks = BLOCK_RE.findall(path.read_text(encoding="utf-8"))
    code = [(lang, body) for lang, body in blocks if lang.lower() in CODE_LANGS]
    count = len(code)
    max_lines = max((body.count("\n") for _, body in code), default=0)
    return count, max_lines


def read_status(path: Path) -> str | None:
    """从 expected_output.md 首行 **状态：…** 取验证状态；无文件返回 None。"""
    if not path.is_file():
        return None
    m = re.search(r"\*\*状态：(.+?)\*\*", path.read_text(encoding="utf-8"))
    if not m:
        return None
    text = m.group(1).strip().rstrip("。").replace("`", "")
    if "已实测" in text:
        return "已实测"
    if "未实测" in text:
        return "未实测"
    if "不适用" in text:
        return "不适用"
    return text or None


def read_prereq(path: Path) -> list[str]:
    """从 README.md 的「## 前置」小节抽取被引用的篇号（如 046）。"""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^## 前置\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m:
        return []
    return sorted(set(re.findall(r"\b(\d{3})\b", m.group(1))))


def repo_dir_of(num: int) -> str | None:
    """篇号对应的仓库文章目录（相对根，含章节前缀），不存在则返回 None。"""
    for d in (REPO_ROOT / "examples").glob(f"*/{num:03d}-*"):
        if d.is_dir():
            return str(d.relative_to(REPO_ROOT))
    return None


def code_files_of(repo_dir: str | None) -> list[str]:
    if not repo_dir:
        return []
    d = REPO_ROOT / repo_dir
    return sorted(f.name for f in d.iterdir() if f.suffix in CODE_SUFFIXES)


def collect(ingest_status: dict[int, str] | None = None) -> list[dict]:
    """汇总全部 81 篇的结构化字段。

    ingest_status: 可选，{篇号: 入库状态} 策展字段；缺失时置 "—"。
    """
    ingest_status = ingest_status or {}
    rows = []
    for num, slug, path in iter_blog_articles():
        ch, ch_name, ch_dir = chapter_of(num)
        repo_dir = repo_dir_of(num)
        verify = read_status(REPO_ROOT / repo_dir / "expected_output.md") if repo_dir else None
        rows.append({
            "id": num,
            "slug": slug,
            "title": article_title(path),
            "chapter": ch,
            "chapter_name": ch_name,
            "chapter_dir": ch_dir,
            "blog_path": str(path.relative_to(REPO_ROOT)),
            "repo_dir": repo_dir,
            "ingest_status": ingest_status.get(num, "—"),
            "verify_status": verify,
            "mechanisms": MECHANISMS.get(num, []),
            "deps": read_prereq(REPO_ROOT / repo_dir / "README.md") if repo_dir else [],
            "block_count": block_stats(path)[0],
            "max_block": block_stats(path)[1],
            "code_files": code_files_of(repo_dir),
        })
    return rows
