# 配图来源说明

`blog/assets/` 下有 57 张图，来源分两类。**这张表决定它们适用哪种许可**：

- **第三方素材**（下表 8 张）：著作权不属于本仓库，**不在 CC BY 4.0 内**；
- **作者自制**（其余 49 张）：由作者截图或绘制，按 CC BY 4.0 授权。

## 第三方素材清单

以下 8 张与 NVIDIA CUDA Programming Guide 的插图**字节完全一致**（md5 比对得出），
判定为 NVIDIA 官方图：

| 本仓库文件 | 官方文档中的名字 |
|---|---|
| `figure-bank-structure-32x32.jpg` | `shared-memory-bank-32x32` |
| `figure-bank-structure-32x33.jpg` | `shared-memory-bank-padded-32x33` |
| `figure-coalesced-access.jpg` | `coalesced-access` |
| `figure-cpu-gpu-interconnect.jpg` | `host-device-execution` |
| `figure-cuda-streams-overlap.jpg` | `async-concurrent-execution` |
| `figure-shared-memory-strided.jpg` | `shared-memory-bank-strided` |
| `figure-thread-blocks-on-sm.jpg` | `thread-blocks-on-sm` |
| `figure-uncoalesced-access.jpg` | `uncoalesced-access` |

出处：NVIDIA, *CUDA C++ Programming Guide*，<https://docs.nvidia.com/cuda/cuda-c-programming-guide/>。
这些图在教程中用于**说明机制**（合并访存、bank 冲突、异步重叠等），不作商业再分发。

## 怎么处理

**使用本仓库内容时：**

- 上表 8 张图**不是** CC BY 4.0 素材，不能仅凭本仓库的许可转载；
- 需要引用时请回到 NVIDIA 原始文档，按其条款使用；
- 其余配图按 [CC BY 4.0](../../docs/LICENSE.md) 使用，署名为本仓库。

**向本仓库贡献配图时：**

- 只用自己绘制或截图的内容，不要贴官方文档的图；
- 若确实需要官方图，在 `NOTICE` 表里补一行来源，**不要**把它算进 CC BY。

## 待办

- 若后续把上述 8 张替换为自绘示意图，请同步删除本表对应行，并在 `docs/LICENSE.md`
  去掉「第三方素材」小节的例外说明。
- 本表是唯一来源：`blog/assets/` 里出现未登记的可疑第三方图，按 bug 处理。
