"""能在本机跑通的脚本做冒烟测试：退出码 0 + 关键输出行。

只覆盖**不需要 NVIDIA GPU** 的脚本。依赖不全时用 skip 而不是 fail——
本仓库的默认环境是“只有标准库 + numpy”，其他层按需安装。

CUDA 脚本不在这里跑：它们的证据必须写进各篇的 expected_output.md。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (相对路径, 参数, 必须出现在 stdout 的片段, 需要的模块)
CASES = [
    ("examples/05-parallel-distributed/044-gpu-task-dispatch/gpu_task_dispatch.py", [], "earliest_finish", ()),
    ("examples/05-parallel-distributed/051-parallel-group-check/parallel_groups_check.py", [], "PASS: split gradients match", ()),
    ("examples/01-gpu-programming/011-tiled-matmul-shared-memory/validate_tiled_index.py", [], "全部断言通过", ()),
    ("examples/01-gpu-programming/019-fluid-grid/fluid_cpu_ref.py", [], "有限性与非负检查通过", ()),
    ("examples/05-parallel-distributed/048-tensor-parallel-linear/tp_split_check.py", [], "TP 两层结果: [12, 6]", ()),
    ("examples/09-medical-imaging/077-sliding-window-toy/sliding_window_toy.py", [], "PASS", ()),
    ("examples/09-medical-imaging/079-ct-amp-check/fp16_numpy_check.py", [], "仅验证格式舍入", ()),
    ("examples/09-medical-imaging/076-ct-intensity-map/intensity_map_cupy.py",
     ["--cpu-only", "--shape", "16", "16", "16", "--repeats", "2"], "CPU检查通过", ()),
    ("examples/02-model-compression/023-gap-vs-fc-distillation/inspect_student_heads.py", [], "以上只是结构检查", ("torch",)),
    ("examples/08-tensorrt-deploy/073-pruning-no-speedup/compact_model.py", [], "已导出模型", ("torch", "onnx")),
    ("examples/08-tensorrt-deploy/067-onnx-export-alignment/export_and_check.py", [], "已保存输入与参考输出",
     ("torch", "onnx", "onnxruntime")),
    ("examples/09-medical-imaging/078-ct-one-step-train/one_step_train.py", [], "只完成一次更新", ("torch", "monai")),
]


def missing(modules: tuple[str, ...]) -> list[str]:
    return [m for m in modules if importlib.util.find_spec(m) is None]


class TestCpuSmoke(unittest.TestCase):
    def test_cpu_scripts_run(self):
        ran = 0
        skipped = []
        for rel, args, marker, requires in CASES:
            path = REPO / rel
            with self.subTest(script=rel):
                self.assertTrue(path.is_file(), f"脚本不存在：{rel}")
                absent = missing(requires)
                if absent:
                    skipped.append(f"{rel}（缺 {', '.join(absent)}）")
                    continue
                result = subprocess.run(
                    [sys.executable, path.name, *args],
                    cwd=path.parent, capture_output=True, text=True, timeout=300,
                )
                self.assertEqual(result.returncode, 0, f"{rel} 退出码非 0\n{result.stderr[-1500:]}")
                self.assertIn(marker, result.stdout, f"{rel} 输出里没有 {marker!r}")
                ran += 1
        if skipped:
            print(f"\n跳过 {len(skipped)} 个（依赖未安装）：" + "、".join(skipped))
        self.assertGreater(ran, 0, "一个脚本都没跑，检查解释器环境")

    def test_gpu_scripts_are_not_in_smoke_list(self):
        """防止有人把需要显卡的脚本塞进 CPU 冒烟列表。"""
        for rel, *_ in CASES:
            text = (REPO / rel).read_text(encoding="utf-8")
            with self.subTest(script=rel):
                self.assertNotIn("import pycuda", text)
                self.assertNotIn("import tensorrt", text)
                self.assertNotIn(".cuda()", text)


if __name__ == "__main__":
    unittest.main()
