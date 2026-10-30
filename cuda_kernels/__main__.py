"""命令行入口：python -m cuda_kernels <命令>

    python -m cuda_kernels list              # 全部篇目与代码文件数
    python -m cuda_kernels info 011          # 某一篇：目录、文件、前缀依赖
    python -m cuda_kernels locate 011        # 只打印目录路径（便于 cd $(...)）
    python -m cuda_kernels run gpu_task_dispatch   # 在该篇目录下运行脚本
"""

from __future__ import annotations

import argparse
import sys

from . import _catalog, articles, files, locate, run


def cmd_list(_args: argparse.Namespace) -> int:
    print(f"{'篇':>4}  {'章':>2}  {'文件':>2}  {'目录'}")
    for a in articles():
        print(f"{a.id:03d}  {a.chapter:>2}  {len(a.files):>2}  {a.dir.relative_to(_catalog.REPO_ROOT)}")
    print(f"\n共 {len(articles())} 个篇目目录")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    a = _catalog.find(args.key)
    print(f"篇号：{a.id:03d}")
    print(f"目录：{a.dir.relative_to(_catalog.REPO_ROOT)}")
    print(f"章节：第 {a.chapter} 章（{a.chapter_dir}）")
    print("文件：")
    for p in a.files:
        print(f"  {p.name:40s} {p.stat().st_size:>7d} B")
    for p in a.files:
        if p.suffix == ".py":
            from .runtime import has_main_guard
            mark = "有" if has_main_guard(p) else "无（import 会跑演示）"
            print(f"  {p.stem:40s} __main__ 守卫：{mark}")
    return 0


def cmd_locate(args: argparse.Namespace) -> int:
    print(locate(args.key))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    return run(args.name, args.args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cuda_kernels", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出全部篇目").set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="查看某一篇")
    p_info.add_argument("key", help="篇号（011）或 slug")
    p_info.set_defaults(func=cmd_info)

    p_loc = sub.add_parser("locate", help="打印某一篇的目录路径")
    p_loc.add_argument("key", help="篇号（011）或 slug")
    p_loc.set_defaults(func=cmd_locate)

    p_run = sub.add_parser("run", help="运行某一个脚本")
    p_run.add_argument("name", help="模块名（文件名去后缀）")
    p_run.add_argument("args", nargs=argparse.REMAINDER, help="透传给脚本的参数")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
