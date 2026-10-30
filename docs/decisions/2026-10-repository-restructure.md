# 结构评估：博文与代码的组织方式

> **路径提示**：本文写于 2026-10 的重构前后，文中出现的 `01-…09-*/`、`docs/ATLAS.md`、
> `docs/PROVENANCE.json` 等是当时的路径。现行对应关系：章节目录 → `examples/<章>/`，
> 能力矩阵 → `docs/reference/capability-matrix.md`，机器清单 → `docs/maintaining/provenance.json`。


> 评估对象：`cuda-kernels`（《大模型训练与推理加速实战》配套代码仓库）
> 评估日期：2026-10-29
> 评估基线：`main` @ `98aa8f3`（含未提交的 09 章目录搬迁）
> 评估方式：静态测绘 + `make check` 实跑 + 跨目录哈希比对，未涉及 GPU

> **本文件是评估时点的快照**（2026-10-29 上午）。第 3 节的问题描述**保留原样**，以便与
> 第 5 节的处置逐条对照；已完成的项在对应条目上标注 ✅，执行结果汇总见文末
> [「七、执行记录」](#七执行记录2026-10-29)。

---

## 一、总评

**作为「书稿配套仓库」，这套组织优于绝大多数同类项目；作为「CUDA kernel 仓库」，它不合格——因为它交付的是可读脚本，不是可复用算子。**

这两个定位并不冲突，问题只在于仓库目前**同时在向两者承诺**，而目录结构只兑现了前者。

| 维度 | 评分 | 依据 |
|---|---|---|
| 架构清晰度 | 8/10 | 章节目录 + 篇号硬约束 + 三类代码（篇目 / 教学单元 / 工程示例）分离 |
| 一致性保障 | 9/10 | md5 溯源链 + 派生文档 `--check` + 29 个结构测试，全部实跑通过 |
| 可运行性 | 4/10 | 52 篇中 43 篇「未实测」；13 个 `.cu` 无统一构建入口 |
| 可复用性 | 3/10 | 75 个 `.py` 零跨目录 import，`pyproject.toml` 无 package 定义 |
| 可维护性 | 7/10 | 派生文档自动生成是好设计，但 5 份博文副本已发散 |

### 现状测绘（实测数字）

| 项 | 数量 | 说明 |
|---|---|---|
| 博文 | 81 篇 | `blog/NNN-*.md`，001–081 连续无断号 |
| 篇目目录 | 53 个 | 九章 `01-`…`09-` 下 `NNN-短名/` |
| 编程语言分布 | 75 `.py` / 13 `.cu` | `.cu` 中 11 个在 `projects/`，章节目录里只有 1 个 |
| 仓库体积 | 185 MB | `.git` 93 MB，`blog/img` 85 MB |
| 实测状态 | 10 / 52 篇有实跑证据 | 覆盖率 17%，其余 `expected_output.md` 为待填 runbook |
| 自检结果 | `make check` 全绿 | 29 个 unittest 通过，跳过 4 个（缺 torch/onnx 等） |

---

## 二、做对了什么

以下四点不是「还行」，而是**确实比主流仓库强**，重构时应保留。

### 2.1 代码→博文的 md5 反向溯源

每个篇目的 `README.md` 记录「博文文件 + 代码块序号 + md5」：

```
01-gpu-programming/011-tiled-matmul-shared-memory/README.md:52-55
- 博文：011-PyCUDA分块矩阵乘法与共享内存.md
- 代码块 #13，298 行，md5 2dc8aff50e27d449daf30afc61a85f4b → tiled_matmul.py
```

`scripts/build_from_blocks.py` 机械校验，`tests/test_provenance.py` 二次把关。cuda-samples、LeetCUDA 都只有代码，没有「代码与讲解一致性」的契约。这等价于把出版流程里的「校样对账」做成了 CI —— 是本仓库真正的护城河。

### 2.2 验证状态诚实分层

README 的「已收录」表逐篇标注「未实测 / 已实测（CPU）/ 部分实测」，`docs/CURRICULUM.md:173-174` 把「没跑过的数不填」写成纪律而非缺陷。多数教学仓库会直接贴一组无出处的 benchmark 数字，这里宁可留空。

### 2.3 派生文档生成 + 防过期检查

`docs/_data/articles.json` 为唯一数据源 → `ATLAS.md` / `ARTICLE_INDEX.md` 由脚本生成，`Makefile:26-31` 带 `--check`。`make check` 会因文档过期而失败。这是工程化做法，教学仓库里罕见。

### 2.4 双授权分离

代码 MIT（`LICENSE`）、文档与博文 CC BY 4.0（`docs/LICENSE.md`），与 fast.ai 一类「书 + 代码」项目的惯例一致。

---

## 三、结构问题

按严重度排列。**P0 表示会丢内容或让仓库失信，P1 表示结构承诺与实现不一致，P2 表示打磨项。**

### P0-1 仓库有两份，且已经开始分叉

```
/workspace/blogspaces/cuda-kernels                                  ← 停在 4f8dad0，落后 8 个提交
/workspace/blogspaces/cuda/大模型训练与推理加速实战/cuda-kernels      ← 98aa8f3，当前工作副本
```

两份**都没有配置 remote**。外层那份不仅旧，还停在「09 章搬迁前」的目录结构上。任何一次在错目录开工都会造成永久分叉，且没有远端可恢复。

同时内层仓库存在未提交的搬迁：

```
 D 09-medical-imaging/ops/backprojection/v0_naive.cu   （旧位置，已删）
?? projects/09-medical-imaging/ops/backprojection/     （新位置，未跟踪）
```

### P0-2 「能跑」是卖点，但证据覆盖率 17%

52 篇完整程序中 43 篇未实测。CI（`.github/workflows/ci.yml`）跑在 `ubuntu-latest`、无 GPU，29 个测试证明的是**「文件与文档对得上」，不是「代码能跑」**。

这不是诚信问题（README 已明说），是**结构问题**：仓库缺一条「GPU 验证」的落点。`expected_output.md` 目前是自由文本，既无法聚合统计，也无法在 CI 里约束「实测覆盖率不得下降」。

### P0-3 博文有 5 份副本，且已发散

| 目录 | 篇数 | 与 `blog/` 完全同哈希 |
|---|---|---|
| `cuda-kernels/blog/` | 81 | —（基准） |
| `博文/` | 84 | 37 |
| `博文_副本/` | 83 | 11 |
| `正文_markdown/` | 63 | 0 |
| `正文/`（txt 改写稿） | — | — |

仓库内声称「代码与博文逐字一致」（README:24-27），**这条只在仓库内部成立**。仓库外已有 4 份内容不同的散文副本，事实源到底是哪个目录，目前无法判定。

### P1-1 章节轴与代码轴不一致，出现空壳章

`docs/CURRICULUM.md:29-39` 用篇号区间定义九章，但章节目录只放**有代码的篇目**：

| 章 | 篇号范围 | 实际目录数 | 状态 |
|---|---|---|---|
| 1 | 001–019（19） | 17 | 正常 |
| 2 | 020–026（7） | 4 | 偏薄 |
| 3 | 027–035（9） | 2 | 偏薄 |
| **4** | **036–041（6）** | **1（039，且无代码）** | **空壳** |
| 5 | 042–051（10） | 6 | 偏薄 |
| **6** | **052–059（8）** | **1（056）** | **空壳** |
| 7 | 060–066（7） | 7 | 完整 |
| 8 | 067–073（7） | 7 | 完整 |
| 9 | 074–081（8） | 8 | 完整 |

读者按章浏览 `04-edge-inference/` 只能看到一个无代码目录，而实际有 6 篇博文。**这是目录结构与导航承诺的矛盾，不是内容缺失。**

`tests/test_structure.py` 没有「每章至少一个目录」这类不变量，所以空壳章永远不会被测出来。

### P1-2 内容与代码同仓，仓库被图片拖到 185 MB

- 仓库 185 MB，`.git` 93 MB，`blog/img` 85 MB / 56 张
- 单张最大 2.4 MB，10 张超过 2.2 MB
- 图片只被 **10 / 81 篇**引用，71 篇无图
- `blog/_meta/CUDA专栏合集-01-81.md` 2.2 MB，是全文再拼一遍

图片命名为 `orca-paste-1789624961307-76b4.png` 这类粘贴产物，不可检索、不可读、无压缩。对比本工作区另一专栏规范（`figure-gpu-vs-cpu.jpg` 可读命名 + 统一 `image/` 目录），这里明显退化。

### P1-3 没有统一构建入口，且硬编码 `sm_86`

- 13 个 `.cu` 中只有 `projects/09-medical-imaging/CMakeLists.txt` 一套构建系统
- `06-custom-operators/056-*/affine_kernel.cu` 只能靠 README 里的裸 `nvcc` 命令
- `projects/09-medical-imaging/ops/*/*.cu` 全部写死 `nvcc -O3 -arch=sm_86`

`-arch=sm_86` 只生成 SASS、不嵌 PTX，在 H100 / B200 / RTX 50 上会直接报 `no kernel image is available for execution`。这正是 myCudaCode 仓库已经踩过的坑，现在在 `projects/` 里复现。

### P1-4 没有可复用的算子，只有可读的脚本

- 75 个 `.py`，**零跨目录 import**
- `pyproject.toml` 无 `[tool.setuptools]` / `packages`，`pip install -e .` 装不出任何模块
- 不存在 `kernels/` 概念，每个文件都是独立运行脚本

对照主流：

- LeetCUDA：`kernels/<op>/{README.md, op.cu, op.py}`，`op.py` 用 `torch.utils.cpp_extension.load()` JIT，可直接 `import`
- flash-attention / cutlass：交付物是库，benchmark 是附属
- cuda-samples：顶层 CMake，每个 sample 独立可编

本仓库交付的是「能读懂的一段代码」。对书籍这是对的，但必须在 README 把定位说死，否则读者会按 kernel 库的期望来用。

### P1-5 命名双轨，且风格本身不统一

目录用英文 slug（`011-tiled-matmul-shared-memory`），博文用中文长标题（`011-PyCUDA分块矩阵乘法与共享内存.md`），映射只能查 `docs/ARTICLE_INDEX.md`。且 blog 内部命名不统一：

```
026-CUDA稀疏矩阵向量乘法（CSR-SpMV）指南.md                           ← 英文
056-手写CUDA vs PyTorch vs cuBLAS性能对比.md               ← 中文
021-模型缩小四倍推理也会快四倍吗.md           ← 中文 + 下划线后缀
```

### P2 打磨项

- `projects/09-medical-imaging/` 与 `09-medical-imaging/` 同名，区别只能靠 README:86-89 解释，应改名区隔
- 缺 `.gitattributes`（`.cu` / `.md` 行尾）、`CITATION.cff`、issue / PR 模板
- 图片无压缩、无 LFS、无可读命名

---

## 四、与主流组织方式对照

| 维度 | 本仓库 | cuda-samples | LeetCUDA | flash-attention | fast.ai book |
|---|---|---|---|---|---|
| 主组织轴 | **书籍篇号 + 教学章** | 难度分级 | 算子名 | 单一算子 | 章节 notebook |
| 代码与讲解 | **同仓 + md5 溯源** | 只有代码 | 代码 + README | 论文 + 代码 | 书 + notebook |
| 可编译性 | 部分（无顶层 CMake） | 顶层 CMake 全编 | JIT 即装 | `setup.py` | — |
| 可 import | ✗ | ✗ | ✓ | ✓ | ✓（fastai 库） |
| 验证证据 | 文档自述（17% 实测） | 无 | 自带 benchmark | CI + 数值测试 | 无 |
| 派生文档 | **自动生成 + 防过期** | 手写 | 手写 | 手写 | 手写 |

**判断**：本仓库在「可追溯性 / 文档工程化」上明显领先主流，在「可编译 / 可复用 / 可验证」三项落后。这不是失误，而是定位选择——它本质是「书稿的构建系统」，不是「kernel 库」。问题只在于目录结构同时在向两种定位承诺。

一个可对照的结论：主流 kernel 仓库解决的是「怎么用」，本仓库解决的是「怎么讲清楚」。前者靠 API 契约，后者靠溯源契约。**两者都需要，但同一份目录结构只能服务一个主轴**，当前的主轴是前者之外的第三条——书籍顺序，这也是它和所有主流仓库都不像的原因。

---

## 五、优化建议

按投入产出排序。前三组互不依赖，可并行。

### 5.1 立刻做（都是止损，不改变结构）

1. **砍掉一份仓库** ✅ 已完成（2026-10-29）
   保留内层为唯一工作副本。删除前确认外层 `4f8dad0` 是内层 HEAD 的严格祖先（无独有内容、无 stash、工作区干净），
   已归档为 `~/workspace/blogspaces/_archive/cuda-kernels-4f8dad0-20261029.bundle`（229 KB，完整历史，
   还原验证：130 个文件 blob 哈希与原件逐一一致）。外层目录已删除。给内层补 remote 仍待办。

2. **提交未完成的搬迁** ✅ 已完成（2026-10-29）
   `09-medical-imaging/ops` → `projects/ct-reconstruction/ops` 已落地，pending 登记同步更新为
   `projects/ct-reconstruction/`（`docs/PROVENANCE.json`），`make check` 全绿，`make strict` 仍按设计失败。

3. **`sm_86` 改为架构探测** ✅ 已完成（2026-10-29）
   10 个文件、17 处 `-arch=sm_86` 全部改为 `-arch=native`；`backprojection/README.md` 补了
   「为什么不能写死 sm_86」的说明（SASS 不嵌 PTX → H100 报 `no kernel image`）。
   顶层 CMake 默认 `CMAKE_CUDA_ARCHITECTURES=native`。

4. **`projects/09-medical-imaging` → `projects/ct-reconstruction`** ✅ 已完成（2026-10-29）
   目录与 6 个文件里的 22 处引用已同步（含 `PROVENANCE.json`、`README.md`、`CONTRIBUTING.md`、
   `docs/落地实施方案.md`）。

### 5.2 一周内（结构补强）

5. **给空壳章加索引 README** ✅ 已完成（2026-10-29）
   九章各有一个由 `scripts/gen_chapters.py` 生成的 `README.md`：列出本章**全部**篇目
   （含未入库的片段篇）、必读顺序、常见误解、逐篇实测状态。`test_structure.py` 新增不变量
   `test_every_chapter_has_index_readme`，漏列任一篇号即失败。

6. **加顶层 CMake** ✅ 已完成（2026-10-29，**未在 N 卡实跑**）
   顶层 `CMakeLists.txt` 覆盖 6 个单文件教学单元，`CMAKE_CUDA_ARCHITECTURES` 默认 `native`；
   工程示例走 `add_subdirectory`；056 的 torch 扩展显式排除并在章节目录做探针提示。
   三条构建路径与失败排查见 `docs/BUILD.md`。**编写环境无 nvcc，本项未编译验证。**

7. **建 `docs/VERIFY.md`（由 `expected_output.md` 生成）** ✅ 已完成（2026-10-29）
   `scripts/gen_verify.py` 派生 53 行明细 + 分章统计 + 两个口径的覆盖率；README 顶部徽章
   `8 / 53 篇全量跑通（15%）` 由 `test_readme_coverage_matches_verify_board` 守着，写旧了直接失败。
   顺带把 3 篇「两个脚本分开记」的模糊声明改写为可归类的「部分实测」（011/019/077），
   并新增「部分实测」这一档——原来的归一化会把它算成「已实测」，等于虚报 7 篇。

8. **图片减重** ✅ 已完成（2026-10-29）
   `blog/img` 85 MB → `image/` 16 MB。42 张 2 MB 级 PNG 截图用 `sips` 转 JPEG（quality 85、长边 1600px）
   并按 `figure-<主题>-NN.jpg` 重命名，14 张原已可读的图保持原名。56 处引用全部改为 `](../image/...)`，
   校验：0 处失效链接、0 处残留 `img/` 引用。对应关系表见 `image/README.md`。
   **注意**：`.git` 仍是 93 MB，旧图留在历史里，`git gc` 删不掉；要真正回收需重写历史，本轮不做。

9. **收敛博文副本**
   `博文/`、`博文_副本/`、`正文_markdown/` 归档到仓库外的 `_archive/`，`blog/` 为唯一事实源，并在 `CONTRIBUTING.md` 写死「新文章只进 `blog/`」。否则 P0-3 的偏离会持续扩大。

### 5.3 中期（取决于定位，二选一）

10. **决定是否包化** ✅ 已决策并落地（2026-10-29）：**包化，但包是加载层而非算子库**
    建了 `cuda_kernels/`（`__init__` / `_catalog` / `runtime` / `ops` / `__main__`），
    `pyproject` 声明 `packages`，`pip install -e .` 在临时 venv 中验证可用。
    **关键取舍**：包内不复制任何源码，只按路径加载章节目录里的规范文件——一旦复制，
    md5 契约立刻失效（两处副本会各自演化）。因此 `ops.<name>` 的语义是「懒加载该篇脚本」，
    没有 `__main__` 守卫的 10 个文件拒绝加载、`.cu` 拒绝 import 并提示走 nvcc。
    定位与边界写进 `docs/PACKAGE.md`，README 第一段同步表态。

11. **统一 slug**
    `blog/NNN-<与目录同 slug>-<中文标题>.md`，中文标题进 H1 而非文件名。一次批量 rename + 全量重跑 md5 对账即可，越晚做成本越高。

### 5.4 不建议做

- **不要把 `blog/` 拆成独立内容站（Docusaurus / mdBook）**。md5 溯源的价值恰恰来自「文与码同仓、同一次 commit 校验」。拆开后退化成两个仓库靠 CI 互查，维护成本不降反升。
- **不要为凑满 81 篇目录造脚本**。README:264 的「不为凑数造脚本」是对的，用 5.2 第 5 条的索引 README 补导航即可，判定标准不必改动。

---

## 六、结论

结构本身设计得比主流教学仓库更讲究——**溯源、派生文档、诚实标注是三个真正的护城河**。当前问题几乎全部出在执行层漂移：

1. 两份仓库
2. 五份博文
3. 三个空壳章
4. 一个写死的 `sm_86`

这四条里已经有三条在 2026-10-29 当轮修完（见第七节）：**两份仓库**已合并为一份并留 bundle 归档，
**三个空壳章**已由九章索引 README 补上导航，**写死的 `sm_86`** 已改为自动探测。
**五份博文**仍未收敛——它需要你确认归档位置，也是对外开源前必须清掉的最后一项结构问题。

结构问题清零之后，真正决定这个仓库成色的就只剩一件事：**实测覆盖率从 15% 往上走**。
看板已经建好，接下来每跑通一篇就往 `expected_output.md` 写一次，数字自己会动。

---

## 七、执行记录（2026-10-29）

本轮按「安全项 + 图片 + 改名 + 包化」执行，**未触碰需要 GPU 的任何结论**：
`expected_output.md` 里没有一个数字被改动，实测覆盖率仍是 8 / 53（15%）。

### 7.1 逐条状态

| 项 | 内容 | 状态 | 验证方式 |
|---|---|---|---|
| P0-1 | 删除重复仓库（外层 4f8dad0） | ✅ | 先 `git bundle` 归档并克隆还原比对 130 个 blob 哈希，再删 |
| P0-2 | 实测覆盖看板 `docs/VERIFY.md` | ✅ | `gen_verify.py --check` + README 徽章一致性测试 |
| P0-3 | 博文 5 份副本收敛 | ⬜ 未做（需你确认归档位置） | — |
| P1-1 | 九章索引 README | ✅ | `gen_chapters.py --check` + 章内篇号完整性测试 |
| P1-2 | 图片 85 MB → 16 MB，上提 `image/` | ✅ | 56 处链接 0 失效、0 残留 `img/` |
| P1-3 | `sm_86` → `native` + 顶层 CMake | ✅（**未编译验证**） | 17 处引用清零；无 nvcc 无法实编 |
| P1-4 | 定位表态 + 加载层 | ✅ | 13 条包契约测试；venv 内 `pip install -e .` 通过 |
| P1-5 | slug 统一（中英混排） | ⬜ 未做（需全量 rename + 重跑 md5） | — |
| P2 | `projects` 改名、`.gitattributes`、`CITATION.cff` | 部分 ✅ | `projects/ct-reconstruction/` 已就位 |

### 7.2 本轮新增

| 文件 | 作用 |
|---|---|
| `scripts/gen_verify.py` → `docs/VERIFY.md` | 实测覆盖看板（53 行明细 + 分章 + 两个口径） |
| `scripts/gen_chapters.py` → 九个章 `README.md` | 章索引：含未入库片段篇的完整导航 |
| `CMakeLists.txt` + `docs/BUILD.md` | 三条构建路径；默认 `native` 架构 |
| `cuda_kernels/`（5 个模块）+ `docs/PACKAGE.md` | 加载层（不复制源码，保 md5 契约） |
| `tests/test_package.py` | 13 条包契约 |
| `image/` + `image/README.md` | 配图集中存放，文件↔篇目对照表 |

### 7.3 新增的四条不变量（都会让 CI 变红）

1. 每章必须有索引 README，且列出该章**全部**篇号 → 空壳章再也藏不住；
2. README 覆盖率徽章必须等于 `docs/VERIFY.md` 的数字 → 手写数字写旧了直接失败；
3. 模块名在章节目录内必须唯一 → `ops.<name>` 不会指向不确定的文件；
4. `cuda_kernels/` 在溯源检查中被显式排除 → 包内一旦偷偷复制篇目源码，契约仍只认章节目录。

### 7.4 验收

```bash
make check          # 44 个测试全绿（原 29 → +15）
make strict         # 仍失败：pending 14 个文件（设计如此，博文发布后才清零）
```

### 7.5 仍未解决

| 项 | 为什么留着 | 建议时机 |
|---|---|---|
| 博文 5 份副本（`博文/`、`博文_副本/`、`正文_markdown/`） | 归档位置需你定；删错不可逆 | 确认后一次性归档到仓库外 `_archive/` |
| 81 篇文件名统一 slug | 全量 rename 会打散 `docs/ARTICLE_INDEX.md` 与 md5 记账，需单独一轮 | 内容冻结后 |
| 实测覆盖率 15% → 更高 | 只能在 N 卡上跑，本机无 `nvcc` | 每次拿到卡就跑 1–2 篇并写证据 |
| `.git` 仍 93 MB | 旧图在历史里，`git gc` 无效，需 `filter-repo` 重写 | 若要开源，重写一次 |
| 顶层 CMake 未编译验证 | 编写环境无 CUDA Toolkit | 首次在有卡机器 `make build` 后回填 |

## 附录：测绘命令

复现本文所有数字：

```bash
cd cuda-kernels

# 自检（无需 GPU）
make check

# 篇目与代码计数
ls blog/*.md | wc -l                                    # 81
find . -maxdepth 2 -type d -name '[0-9][0-9][0-9]-*' \
     -not -path './projects/*' | wc -l                  # 53

# 章节目录分布
for c in 0*/; do
  printf "%-24s %s\n" "$c" \
    "$(find "$c" -maxdepth 1 -type d -name '[0-9][0-9][0-9]-*' | wc -l)"
done

# 体积
du -sh . .git blog/img                                  # 185M / 93M / 85M

# 外链副本哈希比对（在上级目录执行）
python3 - <<'EOF'
import hashlib, pathlib
h = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
blog = {h(p) for p in pathlib.Path('cuda-kernels/blog').glob('*.md')}
for name in ['博文', '博文_副本', '正文_markdown']:
    files = list(pathlib.Path(name).glob('*.md'))
    print(name, len(files), '同哈希', len({h(p) for p in files} & blog))
EOF

# 架构硬编码排查
grep -rn 'arch=sm_' --include='*.cu' --include='*.md' projects/
```
