# 变更记录

本仓库采用「增量提交」，本文件只记录较大的里程碑与决策，不逐条罗列。

## 2026-10-30 · 仓库更名：cuda-kernels → gpu-accel-in-action

原名只覆盖 81 篇里的 34 篇（第 1/6/7 章的 kernel 内容），另 47 篇讲模型压缩、推理部署、
端侧、分布式与医疗影像；且 `cuda-kernels` 在 GitHub 上是 74 个仓库共用的话题标签，
搜不到自己。

- 目录与仓库名改为 `gpu-accel-in-action`；中文名《大模型训练与推理加速实战》不变；
- `pyproject.toml` 发行名同步，**导入名仍是 `cuda_kernels`**（它确实是在加载 CUDA 源码，
  16 个文件在 import，不动）；
- 生成器里的树根同步，`docs/maintaining/repository-layout.md` 重算。

历史文档（`docs/decisions/`、CHANGELOG 旧条目、博文正文）保留旧名，它们记录的是当时事实。

## 2026-10-30 · 重写历史，仓库从 111 MB 降到 19 MB

配图从 `blog/img/`（原始 PNG，85 MB）压缩迁移到 `blog/assets/` 之后，工作区瘦了，
但历史里的旧 blob 仍可达，`git clone` 依然要下 111 MB。公开前做了一次历史重写：

```bash
git filter-repo --invert-paths --path blog/img --path image
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

- **结果**：`.git` 111 MB → 19 MB，仓库 133 MB → 41 MB，最大对象只剩合集（2.2 MB×3 个版本）；
- **不变**：HEAD 树哈希仍是 `c82f09a8`（工作区内容逐字节一致），31 个提交全部保留；
- **变了**：所有 commit hash 重写（旧 HEAD `685390f` → 新 `b98dc33`）。

重写前的完整 `.git` 备份在仓库外：
`~/workspace/blogspaces/_archive/cuda-kernels-git-20261030.tar.gz`（106 MB）。
需要旧历史时解压到临时目录用 `git --git-dir` 查看即可。

**注意**：若已有他人克隆，重写后会与远端不一致，需重新 clone；本仓库当时尚未建远端，
因此没有影响面。

## 2026-10-30 · 许可边界修正

- `docs/LICENSE.md` 重写：授权范围改成**按目录一一对应**的表格，修正仍指向
  `01-`…`09-`（已迁至 `examples/`）的过期描述；根 `LICENSE` 尾注同步，并声明
  「完整划分只有一处事实源」。
- 新增 `blog/assets/NOTICE.md`：经 md5 比对确认 8 张配图与 NVIDIA CUDA Programming Guide
  的插图字节一致，明确它们**不在 CC BY 4.0 范围内**，列出对照表与使用方式。
- 新增两条测试守住这件事：`test_license_paths_exist`（许可文件点名的路径必须存在）、
  `test_notice_lists_third_party_assets`（NOTICE 登记的图必须存在）。

## 2026-10-30 · 博文文件名改为中文标题

- `blog/` 下 55 篇的英文 slug 文件名改为中文标题（如 `011-pycuda-tiled-matmul-shared-memory-light-tech.md`
  → `011-PyCUDA分块矩阵乘法与共享内存.md`），篇号前缀不变；其余 26 篇原名已含中文，未动。
- 同步更新：篇目 `README.md` 的「出处」链接、`docs/maintaining/provenance.json` 的
  `files[].article`、以及全部派生文档（章索引 / 文章索引 / 能力矩阵）。
- 删除改名脚本遗留的 `blog/rename_map*.json`；旧名可从 `git log --follow` 追溯。

## 2026-10-29 · 结构重构：示例分组、文档分区、契约三分

**目录**（提交 b9a86c9）

- 九章 `01-…09-*/` → `examples/<章>/`：顶层按职责分层（文章 / 示例 / 项目 / 工具 / 文档），
  篇目编号与短名不变。
- 配图 `image/` → `blog/assets/`，与文章同层；56 处正文引用改 `assets/…`。
- `projects/09-medical-imaging` → `projects/ct-reconstruction`，消除与章节目录同名歧义。

**契约**（提交 7bc2351）

- 溯源从二分法（逐字 / 待博文 pending）改为三分法：逐字、拼接、**独立项目 standalone**。
  完整工程按自己的构建与测试验收，不再用「有没有写过文章」当合格标准；
  `make strict` 改为校验 `standalone[].built_by` 声明的构建入口真实存在。
- 新增 `docs/maintaining/provenance.md`：把「内容关联 / 运行验证 / 项目成熟度」
  三个维度写清楚，取代散落在 README 与 CONTRIBUTING 的重复规定。

**文档**（提交 7bc2351）

- `docs/` 按用途分区：`getting-started` / `learning-path` / `reference/` /
  `maintaining/` / `decisions/`，文件名从 `ATLAS` `VERIFY` `SKELETON` 这类内部代号
  改成读者能直接看懂的名字。
- README 改为入门优先：定位 → 快速开始 → 九章路线 → 仓库分区 → 文档导航 → 贡献。
  原首页的 53 行收录表与维护规则移交 `article-index` 与 `provenance`。
- 新增 `docs/STRUCTURE.md` 生成器（现为 `docs/maintaining/repository-layout.md`）与
  `docs/README.md` 导航；全仓相对链接 0 失效。

**校验**

- 测试 29 → 44 条（新增：实测看板、章索引完整性、包契约、独立项目登记）。
- `make check` 全绿；`make strict` 通过。

## 2026-10-28 · P0 正规化（提交 6478c51）

- 新增 `LICENSE`（代码 MIT）与 `docs/LICENSE.md`（文档 CC BY 4.0），补齐双许可。
- 新增 `scripts/corpus.py`：收口 `CHAPTERS` / `MECHANISMS` / `LEGEND` 单一标签词表，
  并提供 `collect()` 自动汇总 81 篇字段。
- 新增 `docs/_data/articles.json`：81 篇机器可读清单（检索与索引的数据源）。
- 新增 `scripts/gen_manifest.py`：生成上述清单（`--check` 可校验）。
- 新增 `scripts/gen_index.py`：由清单生成 `docs/ARTICLE_INDEX.md`
  （只重算映射表，前言逐字保留，消除手工大表漂移）。
- 新增 `scripts/search.py`：按 id / 标题 / 标签 / 章节 / 状态 / 依赖 检索，零三方依赖。
- 新增 `Makefile` + `pyproject.toml`：`make check` 一键校验
  （atlas / index / manifest `--check` + 代码溯源 md5 + 静态检查 + 测试）。
- `gen_atlas.py` 改为复用 `corpus` 的标签词表；`static_check.py` 登记本地模块
  `corpus` / `search`。
- 新增 `tests/test_manifest.py`（9 例）。

## 2026-10-28 · P1 工程脚手架 + 契约确认

- 决策：**保持 md5 契约不变**，代码可读化改动落在 `blog/` 代码块后重抽取
  （见 `CONTRIBUTING.md` 的「核心契约」一节）。
- 新增 `CONTRIBUTING.md`：固化契约与工作流、新增一篇/更新元数据的标准步骤。
- 新增 `.github/workflows/ci.yml`：push/PR 到 `main` 自动跑 `make check`。
- 试点：在 `blog/056-*.md` 的代码块加模块 docstring，重抽取到
  `06-custom-operators/056-custom-op-vs-eager/affine.py`，验证
  `make check` 仍全绿、md5 溯源保持。
