# 授权范围

本仓库有**两套许可**，按内容类型划分，与目录一一对应。下面这张表就是判定标准，
不按「文件长得像代码还是像文章」猜。

| 内容 | 位置 | 许可 |
|---|---|---|
| **代码与构建配置** | `examples/`、`projects/`、`cuda_kernels/`、`scripts/`、`tests/`、`.github/`、`CMakeLists.txt`、`Makefile`、`pyproject.toml`、`requirements/` | **MIT**，见仓库根目录 [`LICENSE`](../LICENSE) |
| **文档与教程** | `README.md`、`CONTRIBUTING.md`、`CODE_OF_CONDUCT.md`、`SECURITY.md`、`CHANGELOG.md`、`docs/`、`blog/*.md`、`blog/_meta/` | **CC BY 4.0**，见本文件 |
| **配图** | `blog/assets/` | 作者自制部分按 CC BY 4.0；**标注来源的第三方图不在此列**，见下节 |

同一个文件若同时含说明文字与代码（例如 `README.md` 里的代码块），文字按 CC BY 4.0、
代码按 MIT 分别对待。

## CC BY 4.0 是什么

- 许可全文（法律文本）：<https://creativecommons.org/licenses/by/4.0/legalcode>
- 人类可读摘要：<https://creativecommons.org/licenses/by/4.0/deed.zh>

**你可以**：复制、分发、展示、改编、二次创作，包括商业用途。

**你必须**：

- **署名**：注明原作者／出处，建议链接回本仓库与对应教程；
- 若做了修改，以合理方式标明变更；
- 不得用许可人的商标或名义为你的使用背书。

## 第三方素材

`blog/assets/` 里的部分示意图来自 **NVIDIA 官方文档**（CUDA Programming Guide 等）。
这类图的著作权不属于本仓库，因此：

- **不在 CC BY 4.0 范围内**，不会被本仓库再许可；
- 使用它们时请遵循原出处的条款，并保留原始出处；
- 逐张清单见 [`blog/assets/NOTICE.md`](../blog/assets/NOTICE.md)（8 张，经 md5 比对确认）。

仓库正文中引用 NVIDIA 文档的地方一律为**转述与链接**，不整段翻译原文；
数据与结论的来源在每篇末尾的参考资料中给出。

## 授予与例外

- 你可以基于本仓库内容做课程、书籍、内部培训，无需事先许可（遵守上表各自的条款即可）；
- **不允许**用本仓库或其衍生内容冒充官方文档，也不得暗示 NVIDIA 或原作者为你的使用背书；
- 仓库按现状提供，不含任何担保，见 MIT 的免责条款。

## 变更记录

许可范围随目录结构变化而更新。若发现本文件指向已不存在的路径，请提 issue——
`tests/test_structure.py` 会在路径失效时失败，这类过期不该活过一个 commit。
