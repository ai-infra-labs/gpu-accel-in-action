# 读代码的骨架

61 个 Python 文件，来自 53 篇不同的文章，作者是同一个人但写作时间跨度很长。
好在它们被同一条约束按住：**每个文件都必须与博文里的代码块逐字一致**。

这条约束有个副作用必须正视：**不能抽公共库**。计时、分位统计、`grid` 计算、
随机种子这些在 17–28 个文件里各写一遍。这不是疏忽，而是"读者从博文复制走就能跑"
的代价。所以连续性只能放在**结构**上——每个文件都按同一顺序组织。

## 五要素

任何一篇的完整程序，都按这个顺序展开：

| # | 要素 | 在文件里长什么样 | 为什么要单列 |
|---:|---|---|---|
| 1 | **参数区** | 开头的常量块（`N`、`M`、`block_size`、`TILE`）或 `argparse` | 换规模、换设备只需要改这一处 |
| 2 | **计算主体** | `KERNEL_CODE` 字符串 + `SourceModule` / `@cuda.jit` / `torch` 算子 | 讲"为什么这样算"的地方 |
| 3 | **正确性校验** | `np.testing.*` / `torch.testing.*` / `assert`，**永远在计时之前** | 没对上的数，谈快慢没意义 |
| 4 | **计时与输出** | CUDA Event 或 `perf_counter` + `median`/`p95`，只测它声称测的那一段 | 口径比数字重要 |
| 5 | **入口** | `if __name__ == "__main__":`，或明确说明为什么没有 | 决定"能不能被 import"，也决定读者怎么跑 |

现状统计（`scripts/static_check.py` 之外另做的一次扫描）：61 个文件里
**51 个有入口、59 个带校验、38 个带计时**。剩下的不是漏写——是模块或纯检查脚本，
见下面的白名单。

## 三个代表文件怎么读

| 文件 | 读它学什么 |
|---|---|
| `examples/01-gpu-programming/001-fc-forward-backward/fc_forward_backward.py` | 最小骨架：参数区 → 前向/`dX`/`dW` 三个 kernel → 两个 Event 窗口 → NumPy 三组对照。看完这一个，后面 30 篇都是它的变体 |
| `examples/05-parallel-distributed/042-cuda-streams-concurrency/streams_demo.py` | 并发的骨架在于**同步点的位置**：同样的工作，`synchronize` 放在循环里还是循环外，结论完全不同 |
| `examples/08-tensorrt-deploy/067-onnx-export-alignment/export_and_check.py` | 校验最完整的一篇：13 组用例、四层检查、容差公式都写出来了。想知道"什么叫验证过"，照它抄 |

## 十个没有入口的文件（白名单）

| 文件 | 为什么没有 `__main__` |
|---|---|
| `001`、`002`、`005`、`006`、`007`、`009`、`011` 的主脚本 | 博文附录就是按模块级组织的，顺序执行；加 `main()` 会与博文不一致 |
| `examples/05-parallel-distributed/051-parallel-group-check/parallel_groups_check.py` | 断言与打印都在模块级，本身就是检查脚本 |
| `examples/06-custom-operators/056-custom-op-vs-eager/affine.py` | 被 `bench_three_ways.py` import 的扩展加载模块 |
| `examples/09-medical-imaging/079-ct-amp-check/fp16_numpy_check.py` | 纯格式检查，没有可调参数 |

改这份白名单要同时改 `tests/test_skeleton.py` 的 `NO_MAIN`，并在这里写理由——
否则测试会失败。

## 新增一篇时对着这张表检查

1. 五要素齐不齐？缺的那一条能不能说清为什么缺（写进本文件的白名单）
2. 校验是否在计时**之前**？先校验再基准，是全书统一的次序
3. 打印的每一个数，是否都能在 `expected_output.md` 里找到对应字段？
4. README 是否七个段落齐（定位 / 前置 / 文件 / 运行 / 环境 / 观察点 / 验证 / 出处）？
5. 例子是不是"最小可解释"——去掉注释后还能看懂在算什么吗？

## 测试盯的就是这几条

| 测试 | 断言 |
|---|---|
| `tests/test_skeleton.py` | 每个代码文件有参数区、校验、输出，入口或白名单 |
| `tests/test_structure.py` | 每个目录有 README 七段、README 与进度表状态一致 |
| `tests/test_provenance.py` | 每个文件都能追溯到博文代码块（逐字或登记拼接） |

三条都不需要显卡。GPU 上的数字只有各篇 `expected_output.md` 算数——
这也是为什么仓库里 43 篇的 `expected_output.md` 是待填 runbook，而不是空白模板。
