# `cuda_kernels`：加载层，不是算子库

## 它解决什么问题

81 篇里有 53 个代码目录、61 个 Python 脚本。想跑第 44 篇的调度演示，读者要自己
沿着 `05-parallel-distributed/044-gpu-task-dispatch/` 找进去；想在 notebook 里
调用某个 kernel，只能 `sys.path.append` 加相对路径。

这个包把「按名字找到、加载、运行」变成两行：

```python
import cuda_kernels as ck

ck.locate("011")                     # -> .../examples/01-gpu-programming/011-tiled-matmul-shared-memory
ck.files(11)                         # -> (tiled_matmul.py, validate_tiled_index.py)
ck.load("gpu_task_dispatch").main()  # 载入为模块并调用入口
ck.run("gpu_task_dispatch")          # 等价于在该目录下跑脚本，返回退出码
```

```python
from cuda_kernels import ops
ops.gpu_task_dispatch            # 懒加载，取属性时才 import
ops.describe()                   # 打印「模块名 → 篇目」对照
```

命令行同样可用：

```bash
python -m cuda_kernels list          # 53 个篇目目录
python -m cuda_kernels info 011      # 文件、行数、是否带 __main__ 守卫
python -m cuda_kernels locate 061
python -m cuda_kernels run gpu_task_dispatch
```

## 它刻意不是什么

**不是可 import 的算子库。** 即使 `pip install -e .` 装上，你拿到的也只是
「查找 + 加载 + 运行」的能力，不是 61 个脚本的集合。

原因只有一条：**仓库的硬规矩是代码与博文逐字一致**，靠 md5 对账
（`scripts/build_from_blocks.py`）。如果为了做成库，把 `tiled_matmul.py` 复制一份进
`cuda_kernels/ops/`，就会出现两个会各自演化的副本：

- 改了包里的那份，博文里的那份就旧了，读者复制博文代码拿到的是旧版本；
- 改了博文那份，包里的那份就旧了，且 md5 校验会失败。

两条路都会让「逐字一致」这张纸被戳破。所以包只做**视图**：源文件永远只有章节目录
那一份，包内一行源码都不复制。

这个取舍也解释了为什么 `ops.<name>` 有时会报错：

| 情况 | 行为 |
|---|---|
| 文件有 `if __name__ == "__main__"` 守卫 | 正常加载，只执行 import 与函数定义 |
| 没有守卫（10 个文件，如 001/011/051） | 拒绝加载，提示改用 `ck.run()`——import 它们等于跑一遍演示 |
| `.cu` 文件 | 拒绝加载，提示走 `nvcc` / CMake（见 [BUILD.md](../getting-started.md)） |
| 缺第三方依赖（pycuda / torch / TensorRT） | 抛出原样的 `ImportError`，不吞 |

## 安装

包需要与章节目录同级（它靠 `Path(__file__).parent.parent` 找仓库根），所以只有
editable 安装有意义：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -c "import cuda_kernels as ck; print(len(ck.articles()))"   # 53
```

不想装包也可以，在仓库根目录直接 `python3 -m cuda_kernels ...` 或
`PYTHONPATH=. python3 your_script.py`。

> 系统 Python 直接 `pip install -e .` 会被 PEP 668 挡下（externally-managed-environment），
> 用 venv，或加 `--break-system-packages`（不推荐）。

## 边界

- **只扫描章节目录**（`0[1-9]-*/NNN-*`）。`projects/ct-reconstruction/` 的教学单元不在
  目录里，因为它们不是 81 篇的一部分，且构建方式不同（CMake，见 BUILD.md）。
- **不缓存 import 结果到磁盘**。每次进程重新 import 都重新执行模块级代码，这是 Python 的语义，
  不额外做手脚。
- **加载会临时把源文件目录加进 `sys.path`**，以便同目录脚本互相 import（如 070 复用 069）。
  导入结束后还原，不污染调用方。
- **不替你做 GPU 检查**。`load()` 成功只代表导入没报错，不代表这个 kernel 能跑、
  更不代表它跑对了。实测状态只认 `expected_output.md`（见 [VERIFY.md](verification-status.md)）。

## 自检

```bash
python3 -m unittest tests.test_package -v    # 13 条契约：目录一致、模块名唯一、load 拒绝副作用
```

`tests/test_package.py` 守的三条：

1. 目录扫描结果 = 磁盘真实内容（不维护第二份清单）；
2. 模块名唯一——重名会让 `ops.x` 指向不确定的文件，入库时就得挡住；
3. `load()` 不会在 import 时把演示代码跑起来。
