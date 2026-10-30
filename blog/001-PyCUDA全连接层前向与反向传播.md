# GPU 不认识矩阵：它怎样用 1024 个线程算完一个线性层？

![](assets/figure-fc-forward-01.jpg)

---

先看一个小任务：一家公司要按同一套评分规则，给 1024 个用户分别打分，每个用户有 512 项特征。放在 Python 里，这是一次矩阵—向量乘法：`input` 有 `N` 行（每个用户一行）、`M` 列，`weights` 是一个长度 `M` 的向量，写成一行：

```python
output = input @ weights
```

看起来只是一行矩阵运算。

但把它交给 GPU 后，问题突然多了起来：

- 1024 个用户，应该启动多少个线程？
- `row` 为什么能决定一个线程计算哪个用户？
- `input[row * M + col]` 到底在访问什么？
- 前向传播算出了什么？
- 反向传播里的 `grad_input` 和 `grad_weights` 有什么区别？
- 怎样验证算出来的梯度确实是正确的？

先在心里放一幅小画，后面会反复用到：**GPU 线程是“核算员”，一个用户就是核算员手头的一张表，`weights` 是全员共用的评分规则，`output` 是评完后的最终得分。**

这份 PyCUDA 教学程序非常适合入门，因为它把神经网络中最核心的三件事放到了一起：**线性层、GPU 并行和链式法则**。

我们不会一开始就铺开整页代码，而是「先拆解、后合拢」：逐段讲清线程编号、二维寻址、前向与反向内核，文末再给出**修正后的完整可运行代码**。

但先预警一个结论：这份程序埋了不止一个坑——回读结果时用错 API 会直接报错，`grad_input` 并不是真正的输入梯度，`grad_weights` 的并发累加还存在数据竞争。

本文不只带你读懂它们，还会给出数学推导、正确实现和验证方法。

---

## 先给结论：这是一个单输出、无 bias 的线性层

这份程序的前向计算是：

$$y_n = \sum_{m=0}^{M-1} X_{n,m}w_m$$

其中：

- $X$ 的形状是 $N \times M$；
- $w$ 的形状是 $M$；
- $y$ 的形状是 $N$。

它把每条样本的 $M$ 个特征与同一个权重向量做点积，最终每条样本只输出一个数。

精确地说，这是一个 `**out_features=1`、`bias=False` 的线性层**：`out_features=1` 表示每条样本只输出一个结果，`bias=False` 表示参数里没有偏置项。PyTorch 的 `nn.Linear` 允许任意 `out_features` 并支持关闭 bias，所以它并不是“伪”全连接层，而是多输出全连接层的一个合法特例——只是刻意去掉了偏置、激活和多输出这几个扩展项。

若把它扩展为常见的多输出形式，权重通常是 $M \times K$，输出则是 $N \times K$：

$$Y = XW + b$$

本文先围绕这份程序的单输出版本讲清原理，因为它已经包含了全连接层最关键的计算骨架。

---

## 一张图看懂：CPU 准备任务，GPU 并行处理样本

![](assets/figure-fc-forward-02.jpg)

为了避免把下面这份程序误当成完整训练代码，先亮出它的能力边界：


| 能力             | 当前代码是否实现               |
| -------------- | ---------------------- |
| 单输出线性层前向计算     | 是                      |
| 当前层输入梯度 `dX`   | 是（修正版实现）               |
| 当前层权重梯度 `dW`   | 是（修正版实现）               |
| 真正的损失函数        | 否，用随机 `grad_output` 模拟 |
| 多输出线性层         | 否                      |
| 多层反向传播编排       | 否                      |
| 激活函数反向传播       | 否                      |
| bias 与 bias 梯度 | 否                      |
| 优化器更新权重        | 否                      |
| 完整训练循环         | 否                      |


下面这张表里"是"的部分，就是本文要逐行讲清的东西。

这里最重要的并行映射是：

```text
GPU 线程 0  → 样本 0 → output[0]
GPU 线程 1  → 样本 1 → output[1]
GPU 线程 2  → 样本 2 → output[2]
...
GPU 线程 1023 → 样本 1023 → output[1023]
```

线程不是凭空知道自己应该处理哪条样本。它依靠下面这一行计算全局编号：

```cpp
int row = threadIdx.x + blockIdx.x * blockDim.x;
```

接下来，我们把这行代码彻底拆开。

---

## `row` 从哪里来：Block、Thread 与全局线程编号

![](assets/figure-fc-forward-03.jpg)

程序设置：

```python
block_size = 256
grid_size = (N + block_size - 1) // block_size
```

当 `N = 1024` 时：

```text
grid_size = (1024 + 256 - 1) // 256 = 4
```

于是 GPU 启动 4 个 Block，每个 Block 有 256 个线程，总计 1024 个线程。

```text
Block 0：线程 0～255    → row 0～255
Block 1：线程 0～255    → row 256～511
Block 2：线程 0～255    → row 512～767
Block 3：线程 0～255    → row 768～1023
```

以 Block 2 中的第 7 个线程为例：

```text
threadIdx.x = 7
blockIdx.x  = 2
blockDim.x  = 256

row = 7 + 2 × 256 = 519
```

因此，这个线程负责第 519 条样本。

### 为什么还要判断 `row < N`

这次 `N=1024` 恰好能被 256 整除，但真实数据往往没有这么整齐。

假如 `N=1000`，仍然需要启动 4 个 Block，也就是 1024 个线程。多出来的 24 个线程没有对应样本，所以内核必须写：

```cpp
if (row < N) {
    // 只有合法线程才能访问数组
}
```

这不是可有可无的“保险代码”，而是避免越界访问的必要条件。

---

## 前向传播：一个线程完成一条样本与权重的点积

先对齐一下表达：PyTorch 里 `nn.Linear(M, 1)` 的实际权重是 `1×M` 的行矩阵，公式写作 $y=XW^T+b$；本文为了贴合"一个线程看一行、逐特征乘加"的视角，把权重展平成长度 $M$ 的向量 $w$，写作 $y=Xw$，两者数学等价。

核算员登场。每个线程负责给一个用户打分，做法是把该用户的 512 项特征逐一乘以对应的评分权重再累加。原始前向内核如下：

```cpp
__global__ void forward(
    float *input,
    float *weights,
    float *output,
    int N,
    int M)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < N) {
        float sum = 0.0f;

        for (int col = 0; col < M; col++) {
            sum += input[row * M + col] * weights[col];
        }

        output[row] = sum;
    }
}
```

表面上只有几行，里面却包含了 CUDA 入门最值得掌握的五个概念。

### 1. `__global__`：这个函数由 CPU 发起、在 GPU 上执行

```cpp
__global__ void forward(...)
```

`__global__` 表示这是一个 CUDA Kernel：

- 调用方是 CPU；
- 执行方是 GPU；
- 同一个函数会被许多 GPU 线程同时执行。

Python 端的这次调用：

```python
forward_kernel(
    input_data_gpu,
    weights_gpu,
    output_data_gpu,
    np.int32(N),
    np.int32(M),
    block=(256, 1, 1),
    grid=(4, 1),
)
```

不是只执行一次 `forward`，而是启动 1024 个线程，每个线程都执行一遍同样的内核代码，只是它们看到的 `threadIdx.x` 和 `blockIdx.x` 不同。

### 2. `float *input`：传入的是 GPU 显存地址

Python 中：

```python
input_data_gpu = cuda.to_device(input_data)
```

会在 GPU 上分配显存，并把 CPU 内存中的 `input_data` 复制过去。传给 Kernel 的 `input_data_gpu`，本质上代表设备端内存地址。

因此，内核中的：

```cpp
float *input
```

指向的不是 NumPy 数组对象，而是一段 GPU 可访问的连续显存。

### 3. 二维矩阵为什么可以用一维下标访问

NumPy 中的 `input_data` 形状是：

```text
N 行 × M 列
```

连续的 C-order 数组在内存中按行存放：

```text
第 0 行：[X00, X01, X02, ... X0(M-1)]
第 1 行：[X10, X11, X12, ... X1(M-1)]
第 2 行：[X20, X21, X22, ... X2(M-1)]
```

摊平成一维后：

```text
[X00, X01, ..., X0(M-1), X10, X11, ..., X1(M-1), ...]
```

要找到第 `row` 行、第 `col` 列，需要先跳过前面的 `row` 行。每行有 `M` 个元素，所以偏移量是：

```text
row * M + col
```

于是：

```cpp
input[row * M + col]
```

就是数学中的 $X_{row,col}$。

例如 `M=4`，访问第 2 行第 3 列：

```text
index = 2 × 4 + 3 = 11
```

### 4. `for (col)`：GPU 并行的是样本，不是特征

```cpp
for (int col = 0; col < M; col++)
```

这段实现采用的是：

```text
样本之间：并行
一条样本内部的 M 个特征：串行
```

也就是说，程序为 1024 条样本启动了 1024 个逻辑线程；GPU 会根据 SM 数量和资源占用，把这些线程块并发或分批调度，并不是 1024 个线程必然在同一物理时刻全部执行。对每一条样本而言，一个线程仍要依次执行 512 次乘加。

对第 `row` 条样本：

```text
sum = X[row,0] × w[0]
    + X[row,1] × w[1]
    + ...
    + X[row,M-1] × w[M-1]
```

这就是向量点积。

### 5. `output[row] = sum`：每个线程写自己的结果

每个合法线程只写一个不同的输出位置：

```cpp
output[row] = sum;
```

线程 0 写 `output[0]`，线程 1 写 `output[1]`。线程之间没有写入同一个地址，因此这里不存在写冲突。

---

## 用一个 2×3 的小例子走完前向传播

假设：

$$
X =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix},
\quad
w =
\begin{bmatrix}
0.1 \\
0.2 \\
0.3
\end{bmatrix}
$$

线程 0 负责第 0 条样本：

$$y_0 = 1\times0.1 + 2\times0.2 + 3\times0.3 = 1.4$$

线程 1 负责第 1 条样本：

$$y_1 = 4\times0.1 + 5\times0.2 + 6\times0.3 = 3.2$$

所以：

$$y = [1.4, 3.2]$$

如果只理解到这里，你已经读懂了原程序的前向部分。但神经网络之所以能“训练”，靠的不是只算出 $y$，而是知道损失应该沿着哪些参数往回传。

---

## 反向传播开始前：先把四个边界钉死

进入推导前，先回答一个多数读者会问的问题：这份代码，到底离"一次完整训练"差几步？

### 边界一：没有损失函数，`grad_output` 是模拟的

完整训练要经历：预测 → 与目标值比较得到损失 → 求导得到梯度 → 更新参数。

本章示例**没有目标值、也没有损失函数**。附录 A 的代码直接随机生成了上游梯度：

```python
grad_output = rng.standard_normal(N).astype(np.float32)
```

这相当于假设损失函数已经把 $\partial L/\partial y$ 算好、传了回来。所谓"反向传播"，在这里只完成其中一个局部：已知 `grad_output`，求本层的 `grad_input` 与 `grad_weights`。它不是完整训练循环。

想看到一个真实的损失来源，最简单的是均方误差（MSE）：

$$\ell_n=\frac{1}{2}(y_n-t_n)^2,\qquad L=\frac{1}{N}\sum_{n=0}^{N-1}\ell_n$$

对 $y_n$ 求导：

$$\frac{\partial L}{\partial y_n}=\frac{y_n-t_n}{N}$$

这条公式就是 `grad_output[n]` 应有的值。用它替换随机生成后，本层的梯度就来自真实损失；但要构成一次完整训练，还需要目标值 `target`、损失值计算、参数更新（`weights -= lr * grad_weights`）和迭代训练循环——这些在本例中仍不包含。

### 边界二：损失是 batch 平均值，`1/N` 只能出现一次

一个 batch 里有 $N$ 条样本，每条样本先有自己的损失 $\ell_n$，训练时通常取这批的平均值作为损失：

$$L=\frac{1}{N}\sum_{n=0}^{N-1}\ell_n$$

这决定了哪些梯度要跨样本求和：

输入是每个样本"私有"的，所以 `grad_input` 各算各的；权重是所有样本"共享"的，所以 `grad_weights` 必须汇总所有样本的贡献。


| 梯度                | 是否跨样本求和 | 原因             |
| ----------------- | -------: | -------------- |
| `grad_output[n]`  | 否       | 每条样本的损失各有自己的导数 |
| `grad_input[n,m]` | 否       | 每个样本有自己的输入特征   |
| `grad_weights[m]` | 是       | 所有样本共享同一份权重    |


如果损失已经用了 batch 平均，$1/N$ 就包含在 `grad_output` 里（见边界一的 MSE 导数），`grad_weights` 的 Kernel 里不要再除一次。

### 边界三：先算完所有梯度，再统一更新参数

常见的误解是"先更新当前层权重，再带着新权重算前一层"。标准训练不是这样：

```text
第一阶段 backward ：从最后一层向第一层，只算梯度，不改任何参数
第二阶段 step     ：用已经算好的梯度，统一更新所有参数
```

框架里对应两行：

```python
loss.backward()      # 所有层的 grad_* 算好
optimizer.step()     # W1 -= lr*grad_W1，W2 -= lr*grad_W2，……
```

所以本示例只完成第一阶段（算出 `grad_weights`），没有 `weights -= lr * grad_weights` 这一步——这是刻意留下的边界，不是遗漏。

### 边界四：`grad_input` 交给前一层，是"下一次 Kernel 调用"

假设一个两层网络：

![](assets/figure-fc-forward-04.jpg)

对第 2 层，$H$ 是它的输入，$\partial L/\partial H$ 是第 2 层的 `grad_input`；对第 1 层，$H$ 是它的输出，$\partial L/\partial H$ 是第 1 层的 `grad_output`。两者是同一个量：

![](assets/figure-fc-forward-05.jpg)

但这个交接不是 Kernel 顺手完成的，而是 CPU 端逐层发起调用。先看清楚形状：一般两层网络是 $X(N\times M)\to H(N\times K)\to Y(N\times 1)$，第 1 层（$M\to K$）有 $K$ 个隐藏单元，所以 $H$ 是矩阵而不是向量。逐层正确的写法是：

![](assets/figure-fc-forward-06.jpg)

通用线性层的输入梯度，是对该层 $K$ 个输出求和：

$$\frac{\partial L}{\partial X_{n,m}}=\sum_{k=0}^{K-1}\frac{\partial L}{\partial H_{n,k}}\,W_{m,k}\quad\Longleftrightarrow\quad dX=dH\,W^T$$

本文实现的 `backward_input` 是这条通用公式在 $K=1$ 时的特例：它假定每条样本只有一个输出，因此 `grad_output[row]` 是标量、一个线程算一个分量即可。对 $K>1$ 的中间层，`grad_output` 是 $N\times K$ 矩阵，这篇文章的 Kernel 不能直接复用，需要按通用公式实现（或交给 cuBLAS 的 GEMM）。

![](assets/figure-fc-forward-07.jpg)

注意：本文示例只有一个线性层，`grad_input` 算完之后没有任何"前一层"来消费它。正文里说"传给前一层"，描述的是它在多层网络中的数学角色，不是本例已经实际发生的动作。

---

## 反向传播先别看代码：先认清三个梯度

![](assets/figure-fc-forward-08.jpg)

设最终损失为 $L$，线性层输出为：

$$y_n = \sum_m X_{n,m}w_m$$

承上文：损失函数（或后一层）会把边界一里说的那个上游梯度传回来：

$$g_n = \frac{\partial L}{\partial y_n}$$

它叫上游梯度，也常写作 `grad_output` 或 `dy`。

从这个上游梯度出发，线性层需要计算两类梯度。

### 输入梯度 `grad_input`

输入 $X_{n,m}$ 对输出 $y_n$ 的局部导数是 $w_m$：

$$\frac{\partial y_n}{\partial X_{n,m}} = w_m$$

根据链式法则：

$$\frac{\partial L}{\partial X_{n,m}} = \frac{\partial L}{\partial y_n} \frac{\partial y_n}{\partial X_{n,m}} = g_n w_m$$

因此：

```text
grad_input[n, m] = grad_output[n] × weights[m]
```

它的形状必须与输入 $X$ 相同，即 $N \times M$。

### 权重梯度 `grad_weights`

权重 $w_m$ 被所有 $N$ 条样本共同使用，所以它的梯度必须汇总所有样本的贡献：

$$\frac{\partial L}{\partial w_m} = \sum_{n=0}^{N-1} \frac{\partial L}{\partial y_n} \frac{\partial y_n}{\partial w_m} = \sum_{n=0}^{N-1} g_nX_{n,m}$$

因此：

```text
grad_weights[m] = Σ grad_output[n] × input[n, m]
```

它的形状与权重 $w$ 相同，即 $M$。

### 三种量不要混为一谈


| 名称             | 数学含义                    | 正确形状    | 作用         |
| -------------- | -----------------------: | -------: | ---------- |
| `output`       | $y$                     | `N`     | 前向结果       |
| `grad_output`  | $\partial L/\partial y$ | `N`     | 从后一层传回来的梯度 |
| `grad_input`   | $\partial L/\partial X$ | `N × M` | 继续传给前一层    |
| `grad_weights` | $\partial L/\partial w$ | `M`     | 用来更新权重     |


这张表正是原代码问题的核心：它创建了长度为 `N` 的 `grad_input`，里面最初装的其实是 `grad_output`，随后又试图把它覆盖为输入梯度。但真正的输入梯度应该有 `N × M` 个元素，容器从一开始就不够大。

---

## 原始反向 Kernel 哪里错了

原代码的核心部分是：

```cpp
float grad = 0.0f;

for (int col = 0; col < M; col++) {
    grad += grad_input[row] * weights[col];
    grad_weights[col] += input[row * M + col] * grad_input[row];
}

grad_input[row] = grad;
```

它有三个层次的问题。

### 问题一：`grad_input` 实际装的是 `grad_output`

Python 端创建它时写的是：

```python
grad_input = np.random.randn(N).astype(np.float32)
```

长度为 `N`，而且它在反向传播开始前就有值。根据计算过程，这些值只能代表：

```text
grad_output[row] = dL/dy[row]
```

所以更准确的变量名应该是 `grad_output`。

### 问题二：代码算出的是一个标量和，不是真正的输入梯度

下面这行被循环执行 `M` 次：

```cpp
grad += grad_input[row] * weights[col];
```

最终得到的是 `g_row` 乘上 `M` 个权重之和：

$$grad = g_{row}\sum_m w_m$$

但真正的输入梯度是一整行向量。用一个数值例子钉死：假设第 `row` 个样本的 `grad_output[row] = 2`，`weights = [0.1, 0.2, 0.3]`，正确的输入梯度应为：

$$[2\times0.1,\ 2\times0.2,\ 2\times0.3] = [0.2,\ 0.4,\ 0.6]$$

第 `row` 个样本有 `M` 个输入特征，就该有 `M` 个输入梯度。`row` 只能定位到样本，`row` 与 `col` 一起才能定位到样本中的某一个特征——当前层有 `M` 个输入，就必须为每个样本保存 `M` 个输入梯度。

原代码却把这三项累加成一个数：

$$0.2 + 0.4 + 0.6 = 1.2$$

然后只写回一个标量：

```cpp
grad_input[row] = grad;
```

并不能得到 $dL/dX$——三行特征梯度被压缩成一个标量，无法再区分每个特征各自承担多少责任。

> **能力边界**
>
> 这个 `backward_input` Kernel 只适用于 `out_features=1` 的单输出线性层：它假定每条样本只有一个输出，因此 `grad_output[row]` 是标量。对一般的多输出中间层（$M$ 输入 → $K$ 输出），输入梯度应为 $dX = dY\,W^T$（对 $K$ 求和），本 Kernel 是 $K=1$ 时的特例。它也没有处理激活函数反向、bias 梯度、参数更新与多层编排。

> **工程深度｜验证状态：** L1 入门｜静态复核 + CPU 侧逻辑验证（未在本机实测 GPU；建议上机编译运行）
### 问题三：`grad_weights[col] += ...` 存在数据竞争

![](assets/figure-fc-forward-09.jpg)

回到核算员的画面：所有线程都在更新同一套 `grad_weights`，相当于 1024 名核算员同时往同一页评分规则上改数字——后写的人会覆盖先写的人。所有样本线程都会执行：

```cpp
grad_weights[col] += input[row * M + col] * grad_input[row];
```

当多个线程同时更新同一个 `grad_weights[col]` 时，`+=` 并不是一个不可分割的动作。它大致包含：

![](assets/figure-fc-forward-10.jpg)

假设旧值是 10：

![](assets/figure-fc-forward-11.jpg)

正确结果应该是 15，最终却可能只剩 12 或 13。这就是数据竞争，也叫丢失更新。

更麻烦的是：程序可能不崩溃，只是每次得到略有不同的错误答案。

### `output` 参数没有被使用

反向内核接收了：

```cpp
float *output
```

但函数体里没有读取它。对纯线性层 $y=Xw$ 而言，计算 `dX` 和 `dw` 本来也不需要保存 `y`，所以这个参数可以删除。

---

## 正确的反向传播应该怎样并行

为了先讲清正确性，可以把反向传播拆成两个 Kernel。

### Kernel 1：计算 `grad_input`

`grad_input` 有 `N × M` 个元素，可以让每个线程负责一个元素：

```cpp
__global__ void backward_input(
    const float *grad_output,
    const float *weights,
    float *grad_input,
    int N,
    int M)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = N * M;

    if (idx < total) {
        int row = idx / M;
        int col = idx % M;
        grad_input[idx] = grad_output[row] * weights[col];
    }
}
```

这里：

![](assets/figure-fc-forward-12.jpg)

### Kernel 2：计算 `grad_weights`

一种容易理解、没有写冲突的实现是：每个线程负责一个权重，并在该线程内部遍历所有样本。

```cpp
__global__ void backward_weights(
    const float *input,
    const float *grad_output,
    float *grad_weights,
    int N,
    int M)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (col < M) {
        float sum = 0.0f;

        for (int row = 0; row < N; row++) {
            sum += input[row * M + col] * grad_output[row];
        }

        grad_weights[col] = sum;
    }
}
```

因为线程 `col` 只写 `grad_weights[col]`，不同线程不会写同一个地址，所以无需 `atomicAdd`。

这两个 Kernel 的映射是：

![](assets/figure-fc-forward-13.jpg)

这是一份“正确性优先”的教学实现——它的目的就是演示怎样用“写入地址独占”消除竞态，不代表生产级权重梯度 Kernel。大规模场景下，`grad_weights` 通常会进一步使用分块归约、共享内存，或者直接交给 cuBLAS 完成；生产实践中，单输出权重梯度走 GEMV 或归约，多输出一般交给 GEMM，更深入的自研高性能内核则涉及 CUTLASS 等分层分块抽象。

---

这里的三个 NumPy 公式值得记住：

![](assets/figure-fc-forward-14.jpg)

它们分别对应：

$$y=Xw$$

$$\frac{\partial L}{\partial X}=g\,w^T$$

$$\frac{\partial L}{\partial w}=X^Tg$$

---

## 为什么不用 `atomicAdd` 直接修补原代码

最小改动确实可以把：

```cpp
grad_weights[col] += contribution;
```

改成：

```cpp
atomicAdd(&grad_weights[col], contribution);
```

这样能避免丢失更新，但它不是这个例子最清晰的教学解法。

因为 `N` 个线程会反复竞争同一批 `M` 个地址。原子操作保证正确性，却可能把热点写入串行化。

还要认清 `atomicAdd` 的边界：它保证每一次更新都不丢失，但不保证浮点结果的逐位确定——线程执行顺序依旧不定，而浮点加法不满足结合律，所以不同运行之间可能只剩末位差异。教学示例里可以忽略，但要知道原子操作不是“确定性”的代名词。

更典型的优化路径是：

![](assets/figure-fc-forward-15.jpg)

所以要区分两个问题：

- `atomicAdd` 能否修正数据竞争？能。
- `atomicAdd` 是否一定是最高效方案？不是。

---

## 原始前向 Kernel 正确，但并不高效

原始前向计算在数学上是正确的，但它的线程映射存在明显的性能局限。

同一个 Warp 中的线程在同一时刻访问：

![](assets/figure-fc-forward-16.jpg)

相邻线程之间相隔 `M` 个 `float`。当 `M=512` 时，地址跨度为：

```text
512 × 4 B = 2048 B
```

这不是理想的连续合并访问模式。代码对单线程来说在顺序读取一整行，但从 Warp 的同一条指令来看，32 个线程读的是相距很远的地址。

如果只是学习，这份实现足够直观；如果追求性能，可以考虑：

1. 让一个线程块协作计算一条或多条样本；
2. 使用共享内存或 Warp Reduction 完成点积归约；
3. 让线程访问方式更利于合并访存；
4. 实际工程中优先使用 cuBLAS 的 GEMV/GEMM；
5. 多输出全连接层常转化为矩阵乘法（GEMM）；在硬件架构、数据类型（FP16/BF16/TF32/FP8）、矩阵尺寸与库选核等条件满足时，cuBLAS/cuBLASLt 才会选用 Tensor Core 内核。

手写 Kernel 的价值主要是理解计算与并行映射；成熟库的价值是把分块、向量化、流水线和硬件特性做到更好。

不同阶段可以参考这条路线：


| 学习目标            | 推荐路径                    |
| --------------- | ----------------------- |
| 线程、索引、梯度的直觉     | 本文手写 PyCUDA Kernel      |
| 数组级 GPU 计算      | CuPy 等 GPU 数组工具         |
| 深度学习训练          | PyTorch / JAX 等框架       |
| 生产级线性代数         | cuBLAS / cuBLASLt       |
| 自定义高性能矩阵 Kernel | CUDA C++、CUTLASS、Triton |


---

## 原程序的计时为什么不是“纯 Kernel 耗时”

原代码大致是这个计时思路：

```python
start_time = time.time()
forward_kernel(...)
output_data_gpu.get(output_data)
end_time = time.time()
```

不过在谈计时之前，要先纠正一个会让原程序直接崩掉的 API 错误：`cuda.to_device()` 返回的是 `pycuda.driver.DeviceAllocation`，它没有 `.get()` 方法——`.get()` 是 `pycuda.gpuarray.GPUArray` 独有的。想用 `.get()` 回读，前面得改用 `gpuarray.to_gpu()`；要么就用 `cuda.memcpy_dtoh(dst, src)` 把结果拷回 CPU（附录 A 的修正版采用的就是后者）。

也就是说，原程序即便没有任何梯度问题，也会在这一行抛出 `AttributeError`，根本到不了“打印结果”。所以下面的计时讨论，默认已经修好了这一行。

CUDA Kernel 通常是异步发射的：CPU 把任务提交给 GPU 后，不必立刻等它执行完。

而无论用 `GPUArray.get()` 还是 `memcpy_dtoh()`，把结果拷回 CPU 的动作都会等待前面的 GPU 工作完成。因此这段时间更接近：

```text
Python 调用开销
+ Kernel 执行
+ GPU → CPU 拷贝
+ 同步等待
```

反向传播计时还包含两次结果回传，所以它与前向耗时也不能直接公平对比。

如果目标是测量设备端 Kernel，应使用 CUDA Event；如果目标是测量业务端端到端延迟，则应明确保留数据传输和同步。两种计时都合理，关键是不要混称。

另外提醒一句：单次 Event 只用来演示 API 用法，不是稳定的性能结论。`SourceModule` 的编译发生在 Event 计时之前，并不计入这里的 Kernel 时间；但首次调用仍可能受上下文初始化、模块加载、GPU 时钟与缓存状态影响，因此需要预热并重复测量。下面的预热与中位数写法适用于正式 benchmark；附录 A 的代码刻意保持单次 Event 演示，只为展示 API 用法、不作性能结论。可靠做法是先预热、再多次测量取中位数或 P95：

```python
# 预热：反复启动几次，让编译和驱动进入稳定状态
for _ in range(20):
    forward_kernel(...)

# 正式测量：多取几轮，用中位数代表稳定值
times = []
for _ in range(100):
    start.record()
    forward_kernel(...)
    end.record()
    end.synchronize()
    times.append(start.time_till(end))

median_ms = np.median(times)
```

---

## 小结：两个梯度分别流向哪里

把边界三、边界四与"三种梯度"表收拢成一句话：

> `grad_input` 是"责任"，继续向网络前方传递；`grad_weights` 是"调整方向"，交给优化器更新当前层参数。

补充三点：

- 方向由负梯度决定：$\partial L/\partial w$ 为正，则 $w$ 应减小；为负则增大；绝对值越大，损失对该权重越敏感。
- 最简单的更新是梯度下降 $w \leftarrow w-\eta\,\partial L/\partial w$；Adam 等现代优化器在此基础上维护动量和二阶统计，但输入始终离不开正确的 `grad_weights`。
- 执行顺序按边界三：先算完所有层的梯度，再统一更新，不在反向过程中边算边改。

---

## 怎样证明梯度真的正确：不要只看“能运行”

CUDA 程序没有报错，只能证明它没有触发可见的运行时故障，不能证明数学结果正确。

至少应做三层验证。

### 第一层：与 NumPy 参考实现对照

```python
np.testing.assert_allclose(gpu_result, cpu_reference)
```

这是最直接、最必要的验证。

### 第二层：检查形状

![](assets/figure-fc-forward-17.jpg)

很多反向传播错误，在形状检查阶段就能发现。

### 第三层：有限差分梯度检查

对某个权重 $w_m$ 加一个很小的扰动 $\varepsilon$：

$$\frac{\partial L}{\partial w_m} \approx \frac{L(w_m+\varepsilon)-L(w_m-\varepsilon)}{2\varepsilon}$$

再把数值梯度与 CUDA 反向传播结果对比。有限差分速度慢，但很适合验证新写的 Kernel。

---

## 这份程序最值得带走的 10 个结论

1. `row = blockIdx.x * blockDim.x + threadIdx.x` 把线程映射到样本。
2. `input[row * M + col]` 是二维行优先数组的一维寻址公式。
3. 原前向 Kernel 采用“一线程一条样本”，样本并行、特征串行。
4. 这是 `out_features=1`、`bias=False` 的线性层，是多输出全连接层的一个合法特例。
5. `grad_output` 是从后一层传回来的 $dL/dy$，不能与 `grad_input` 混用。
6. 真正的 `grad_input=dL/dX` 形状是 `N×M`，元素为 `grad_output[n] × weights[m]`。
7. `grad_weights=dL/dw` 要汇总所有样本，元素为 `Σ input[n,m] × grad_output[n]`。
8. 多线程执行普通 `+=` 更新同一地址会产生数据竞争；能打印数字不等于答案正确。
9. 当前 Kernel 只计算一层的局部梯度，不会自动调用前一层；`grad_input` 只有在被下一次 Kernel 调用当作前一层的 `grad_output` 时，才算真正完成层间梯度传递。
10. 本例没有损失函数、激活函数、优化器和多层编排，是梯度教学程序，不是完整训练程序。

---

## 结语：神经网络的“魔法”，最终会落到索引、乘加与同步

从 Python 看，全连接层可能只是一行：

```python
y = X @ w
```

从 GPU 看，它变成了四个具体问题：

![](assets/figure-fc-forward-18.jpg)

前向传播负责算出结果，反向传播负责回答“结果为什么错、每个输入和参数应承担多少责任”。

而 `grad_input` 与 `grad_weights` 的区别，正是理解反向传播的分水岭：一个把梯度继续传回前一层，一个把调整方向交给优化器。

当你能从形状、公式、线程和内存四个角度同时解释这段代码时，CUDA Kernel 就不再是一串难懂的下标，神经网络也不再是一只黑箱。

---

## 附录 A：修正版完整代码

下面这份代码保留原程序的整体结构，修正了变量语义、梯度形状、并发写冲突和计时方式，并给每个 Kernel 与关键调用加了注释，方便对照正文逐段核对。

```python
"""GPU 不认识矩阵：它怎样用 1024 个线程算完一个线性层？（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy。
"""
import pycuda.driver as cuda

# 自动初始化 CUDA Driver、选择默认 GPU 并创建 CUDA 上下文
import pycuda.autoinit

# 导入 NumPy，用于在 CPU 端生成数组和计算参考答案
import numpy as np

# 导入 SourceModule，用于在运行时编译 CUDA C/C++ 源码字符串
from pycuda.compiler import SourceModule


# 使用原始字符串保存 CUDA C/C++ 源码，避免 Python 转义影响源码内容
kernel_code = r"""
// forward：计算单输出线性层 y = Xw
__global__ void forward(
    const float *input,   // 输入矩阵 X，形状为 N×M，只读
    const float *weights, // 权重向量 w，长度为 M，只读
    float *output,        // 输出向量 y，长度为 N，由 Kernel 写入
    int N,                // 样本数量，也就是输入矩阵的行数
    int M)                // 特征数量，也就是输入矩阵的列数
{
    // 计算当前线程在整个一维 Grid 中的全局编号，并把它作为样本行号
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    // 只让拥有合法样本行号的线程继续执行，避免数组越界
    if (row < N) {
        // 使用寄存器中的局部变量保存当前样本点积的累加结果
        float sum = 0.0f;

        // 依次遍历当前样本的全部 M 个特征
        for (int col = 0; col < M; col++) {
            // 读取 X[row,col] 和 w[col]，完成一次乘法并累加到 sum
            sum += input[row * M + col] * weights[col];
        }

        // 当前线程把最终点积写入自己负责的输出位置
        output[row] = sum;
    }
}


// backward_input：计算输入梯度 dL/dX = grad_output × weights^T
__global__ void backward_input(
    const float *grad_output, // 上游梯度 dL/dy，长度为 N，只读
    const float *weights,     // 前向使用的权重向量 w，长度为 M，只读
    float *grad_input,        // 待写入的输入梯度 dL/dX，形状为 N×M
    int N,                    // 样本数量
    int M)                    // 特征数量
{
    // 为 N×M 个输入梯度元素建立一维全局编号
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 计算输入矩阵和输入梯度矩阵的元素总数
    int total = N * M;

    // 过滤向上取整后多启动的线程，避免越界
    if (idx < total) {
        // 用整数除法把一维编号还原为二维行号
        int row = idx / M;

        // 用取模运算把一维编号还原为二维列号
        int col = idx % M;

        // 根据链式法则计算 dL/dX[row,col] = dL/dy[row] × w[col]
        grad_input[idx] = grad_output[row] * weights[col];
    }
}


// backward_weights：计算权重梯度 dL/dw = X^T × grad_output
__global__ void backward_weights(
    const float *input,       // 前向输入矩阵 X，形状为 N×M，只读
    const float *grad_output, // 上游梯度 dL/dy，长度为 N，只读
    float *grad_weights,      // 待写入的权重梯度 dL/dw，长度为 M
    int N,                    // 样本数量
    int M)                    // 特征数量
{
    // 让每个线程负责一个特征列，也就是一个权重梯度元素
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    // 只让列号位于 [0,M) 内的线程继续执行
    if (col < M) {
        // 在寄存器中保存当前权重收到的全部样本贡献
        float sum = 0.0f;

        // 遍历全部 N 条样本，完成当前列上的归约
        for (int row = 0; row < N; row++) {
            // 累加 X[row,col] × dL/dy[row]
            sum += input[row * M + col] * grad_output[row];
        }

        // 当前线程独占 grad_weights[col]，因此不会发生并发写冲突
        grad_weights[col] = sum;
    }
}
"""


# 调用 NVCC 编译 CUDA 源码，并把结果加载为 PyCUDA 模块
module = SourceModule(kernel_code)

# 从编译后的模块中取得名为 forward 的 Kernel 句柄
forward_kernel = module.get_function("forward")

# 从编译后的模块中取得计算输入梯度的 Kernel 句柄
backward_input_kernel = module.get_function("backward_input")

# 从编译后的模块中取得计算权重梯度的 Kernel 句柄
backward_weights_kernel = module.get_function("backward_weights")


# 设置样本数量：一共有 1024 条输入样本
N = 1024

# 设置特征数量：每条样本包含 512 个 float32 特征
M = 512

# 创建固定种子的随机数生成器，保证每次运行都能复现实验数据
rng = np.random.default_rng(42)

# 生成形状为 N×M 的 float32 输入矩阵 X
input_data = rng.random((N, M), dtype=np.float32)

# 生成长度为 M 的 float32 权重向量 w
weights = rng.standard_normal(M).astype(np.float32)

# 为长度为 N 的前向输出 y 分配 CPU 内存，内容稍后由 GPU 覆盖
output_data = np.empty(N, dtype=np.float32)

# 生成长度为 N 的上游梯度 dL/dy，它是反向传播的输入
grad_output = rng.standard_normal(N).astype(np.float32)

# 为真正的输入梯度 dL/dX 分配 N×M 的 CPU 数组
grad_input = np.empty((N, M), dtype=np.float32)

# 为权重梯度 dL/dw 分配长度为 M 的 CPU 数组
grad_weights = np.empty(M, dtype=np.float32)


# 在 GPU 上分配输入显存，并把 input_data 从 CPU 复制到 GPU
input_gpu = cuda.to_device(input_data)

# 在 GPU 上分配权重显存，并把 weights 从 CPU 复制到 GPU
weights_gpu = cuda.to_device(weights)

# 只为前向输出分配 GPU 显存，因为输出会被 Kernel 完整写入
output_gpu = cuda.mem_alloc(output_data.nbytes)

# 在 GPU 上分配上游梯度显存，并把 grad_output 从 CPU 复制到 GPU
grad_output_gpu = cuda.to_device(grad_output)

# 只为输入梯度分配 GPU 显存，因为全部元素会被 backward_input 写入
grad_input_gpu = cuda.mem_alloc(grad_input.nbytes)

# 只为权重梯度分配 GPU 显存，因为全部元素会被 backward_weights 写入
grad_weights_gpu = cuda.mem_alloc(grad_weights.nbytes)


# 设置每个一维线程块包含 256 个线程
block_size = 256

# PyCUDA 使用三元组描述 Block 的 x、y、z 三个维度
block = (block_size, 1, 1)

# 为 N 个前向任务计算一维 Grid 大小，并使用向上取整避免遗漏样本
grid_n = ((N + block_size - 1) // block_size, 1)

# 为 N×M 个输入梯度元素计算一维 Grid 大小
grid_nm = ((N * M + block_size - 1) // block_size, 1)

# 为 M 个权重梯度元素计算一维 Grid 大小
grid_m = ((M + block_size - 1) // block_size, 1)


# 创建记录 GPU 前向起始时刻的 CUDA Event
forward_start = cuda.Event()

# 创建记录 GPU 前向结束时刻的 CUDA Event
forward_end = cuda.Event()

# 把前向开始事件记录到当前 CUDA Stream
forward_start.record()

# 启动前向 Kernel：每个线程负责一条样本的点积
forward_kernel(
    input_gpu,             # 参数1：GPU 上的输入矩阵
    weights_gpu,           # 参数2：GPU 上的权重向量
    output_gpu,            # 参数3：GPU 上的输出数组
    np.int32(N),           # 参数4：以 int32 形式传入样本数量
    np.int32(M),           # 参数5：以 int32 形式传入特征数量
    block=block,           # 指定每个 Block 的线程组织
    grid=grid_n,           # 指定前向计算所需的 Block 数量
)

# 把前向结束事件排在 Kernel 之后
forward_end.record()

# 等待结束事件完成，确保前向 Kernel 已经执行完毕
forward_end.synchronize()

# 计算两个 Event 之间的设备端时间，单位为毫秒
forward_ms = forward_start.time_till(forward_end)


# 创建记录反向传播起始时刻的 CUDA Event
backward_start = cuda.Event()

# 创建记录反向传播结束时刻的 CUDA Event
backward_end = cuda.Event()

# 把反向开始事件记录到当前 CUDA Stream
backward_start.record()

# 启动输入梯度 Kernel：每个线程负责 dL/dX 中的一个元素
backward_input_kernel(
    grad_output_gpu,       # 参数1：GPU 上的上游梯度 dL/dy
    weights_gpu,           # 参数2：GPU 上的权重向量
    grad_input_gpu,        # 参数3：GPU 上待写入的输入梯度
    np.int32(N),           # 参数4：样本数量
    np.int32(M),           # 参数5：特征数量
    block=block,           # 指定每个 Block 的线程组织
    grid=grid_nm,          # 指定覆盖 N×M 个元素的 Grid
)

# 启动权重梯度 Kernel：每个线程负责一个权重梯度元素
backward_weights_kernel(
    input_gpu,             # 参数1：GPU 上的前向输入矩阵
    grad_output_gpu,       # 参数2：GPU 上的上游梯度 dL/dy
    grad_weights_gpu,      # 参数3：GPU 上待写入的权重梯度
    np.int32(N),           # 参数4：样本数量
    np.int32(M),           # 参数5：特征数量
    block=block,           # 指定每个 Block 的线程组织
    grid=grid_m,           # 指定覆盖 M 个权重的 Grid
)

# 把反向结束事件排在两个反向 Kernel 之后
backward_end.record()

# 等待反向结束事件完成，确保两个反向 Kernel 都执行完毕
backward_end.synchronize()

# 计算两个反向 Kernel 的设备端总耗时，单位为毫秒
backward_ms = backward_start.time_till(backward_end)


# 把 GPU 上的前向输出复制回 CPU 的 output_data
cuda.memcpy_dtoh(output_data, output_gpu)

# 把 GPU 上的输入梯度复制回 CPU 的 grad_input
cuda.memcpy_dtoh(grad_input, grad_input_gpu)

# 把 GPU 上的权重梯度复制回 CPU 的 grad_weights
cuda.memcpy_dtoh(grad_weights, grad_weights_gpu)


# 使用 NumPy 矩阵向量乘法计算前向传播参考答案 y = Xw
output_ref = input_data @ weights

# 利用广播计算输入梯度参考答案 dL/dX[n,m] = dL/dy[n] × w[m]
grad_input_ref = grad_output[:, None] * weights[None, :]

# 使用矩阵向量乘法计算权重梯度参考答案 dL/dw = X^T × dL/dy
grad_weights_ref = input_data.T @ grad_output


# 使用相对和绝对容差检查 GPU 前向输出是否接近 NumPy 结果
np.testing.assert_allclose(
    output_data,
    output_ref,
    rtol=1e-4,
    atol=1e-4,
)

# 检查 GPU 输入梯度是否接近 NumPy 广播计算结果
np.testing.assert_allclose(
    grad_input,
    grad_input_ref,
    rtol=1e-4,
    atol=1e-4,
)

# 检查 GPU 权重梯度是否接近 NumPy 矩阵向量乘法结果
np.testing.assert_allclose(
    grad_weights,
    grad_weights_ref,
    rtol=1e-4,
    atol=1e-4,
)


# 打印前向 Kernel 的设备端执行时间
print(f"前向 Kernel 执行时间: {forward_ms:.3f} ms")

# 打印两个反向 Kernel 的设备端总执行时间
print(f"反向 Kernels 执行时间: {backward_ms:.3f} ms")

# 明确告诉读者三组结果已经通过 CPU 参考实现验证
print("前向输出、输入梯度和权重梯度均通过 NumPy 对照验证")

# 打印第一条样本前 5 个输入特征对应的梯度
print("grad_input[0, :5] =", grad_input[0, :5])

# 打印前 5 个权重对应的梯度
print("grad_weights[:5] =", grad_weights[:5])
```

