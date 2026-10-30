# 文档导航

按「你现在要解决什么问题」找文档，而不是按文件生成方式找。

| 你想做什么 | 看这里 |
|---|---|
| 第一次来，想跑起来 | [快速开始](getting-started.md) |
| 按顺序学，或跳到某一章 | [学习路线](learning-path.md) |
| 查某一篇讲什么、有没有代码 | [文章索引](reference/article-index.md) |
| 查某一篇的依赖、机制、前置 | [能力矩阵](reference/capability-matrix.md) |
| 查某一篇到底跑没跑过 | [验证状态](reference/verification-status.md) |
| 在代码里按名字找/跑某一篇 | [加载层](reference/package.md) |
| 要改仓库，先认清目录 | [仓库结构](maintaining/repository-layout.md) |
| 新增一篇示例，要守什么规矩 | [示例约定](maintaining/example-conventions.md) |
| 搞清「代码凭什么可信」 | [溯源契约](maintaining/provenance.md) |
| 想知道目录为什么这样分 | [结构决策记录](decisions/2026-10-repository-restructure.md) |
| 要转载、二次使用或商用 | [授权范围](LICENSE.md) · [配图来源](../blog/assets/NOTICE.md) |

## 四个分区

```
docs/
├── getting-started.md     入门：环境、第一个示例、编译
├── learning-path.md       九章路线
├── reference/             查阅型（其中 3 个由 make gen 生成）
├── maintaining/           维护者：目录、约定、契约与机器清单
└── decisions/             决策记录：当时的取舍与偏差
```

生成物不要手改：跑 `make gen`，`make check` 会在文档过期时失败。
清单见 [仓库结构 · 第六节](maintaining/repository-layout.md#六docs-按用途分区)。
