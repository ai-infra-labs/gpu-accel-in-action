# cuda-kernels 常用入口
#
#   make check    —— 一键校验：ATLAS / VERIFY / STRUCTURE / 索引 / 清单 / 溯源 / 静态 / 单测 全部一致
#   make gen      —— 重新生成派生文档（ATLAS.md / VERIFY.md / ARTICLE_INDEX.md / articles.json）
#   make test     —— 跑测试套件
#   make strict   —— 发布闸门：独立项目必须声明真实存在的构建入口
#   make search ARGS="nccl"  —— 检索篇目（等同 python3 scripts/search.py）
#   make build    —— 编 projects 下的单文件教学单元（需 CUDA Toolkit，见 docs/BUILD.md）

PY := python3

.PHONY: check gen build atlas verify structure chapters index manifest provenance provenance-strict static test search strict

check: atlas-check verify-check structure-check chapters-check index-check manifest-check provenance static test
	@echo "✅ 全部检查通过"

# manifest（articles.json）必须先生成，atlas/index 都依赖它
gen: manifest atlas verify structure chapters index

atlas:
	$(PY) scripts/gen_atlas.py
verify:
	$(PY) scripts/gen_verify.py
structure:
	$(PY) scripts/gen_structure.py
chapters:
	$(PY) scripts/gen_chapters.py
index:
	$(PY) scripts/gen_index.py
manifest:
	$(PY) scripts/gen_manifest.py

atlas-check:
	$(PY) scripts/gen_atlas.py --check
verify-check:
	$(PY) scripts/gen_verify.py --check
structure-check:
	$(PY) scripts/gen_structure.py --check
chapters-check:
	$(PY) scripts/gen_chapters.py --check
index-check:
	$(PY) scripts/gen_index.py --check
manifest-check:
	$(PY) scripts/gen_manifest.py --check
provenance:
	$(PY) scripts/build_from_blocks.py --check
provenance-strict:
	$(PY) scripts/build_from_blocks.py --check --strict
static:
	$(PY) scripts/static_check.py
test:
	$(PY) -m unittest discover -s tests

# 发布前用这个：独立项目的构建入口必须存在，且全仓代码都要能说清来源
strict: provenance-strict
	@echo "✅ 独立项目的构建入口齐全"

search:
	$(PY) scripts/search.py $(ARGS)

# 需要 CUDA Toolkit；无卡机器上 cmake 配置阶段就会失败，这是预期的
build:
	cmake -B build && cmake --build build -j
