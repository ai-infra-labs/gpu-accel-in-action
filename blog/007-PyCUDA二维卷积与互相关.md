# Kernel 能编译，不代表它在算卷积：拆解 PyCUDA 二维卷积的三个反转

![](assets/figure-conv2d-01.jpg)

## 一段能编译的代码，为什么和 SciPy 对不上？

输入是一张 `512×512` 的图像，卷积核是 `3×3`。每个 GPU 线程负责计算一个输出像素，代码能编译，内核（Kernel）能启动，输出尺寸也完全正确。

然后我把 GPU 结果交给 SciPy 验证：

```python
output_cpu = convolve2d(
    input_data,
    kernel,
    mode='same',
    boundary='fill',
    fillvalue=0
)
```

结果却对不上。不是差在最后几位，而是从方向到数值都可能不同。

更有意思的是，这段代码里的每一行看起来都很合理：

```cpp
value += input[y * input_width + x]
       * kernel[(i + half_k_height) * kernel_width + (j + half_k_width)];
```

输入坐标没越界，卷积核下标也在移动，为什么结果还是错的？

答案不是 GPU 算错了，而是这段名叫 `convolution_2d` 的代码，严格来说根本没有做卷积——它做的是二维互相关（cross-correlation）。

> **本文讨论范围**：单通道、单张图像、`stride=1`、`dilation=1`、奇数尺寸核、零填充（same 模式）。下述所有结论都只在范围内成立，不涉及批处理、多通道、分组卷积与反向传播。

> **工程深度｜验证状态：** L1–L2 入门｜静态复核 + CPU 对比验证（建议上机跑通）
这还只是第一个坑。这篇文章把这段代码反复"反转"三次，顺手拆掉另外四个容易翻车的地方。

---

## 一句话结论

这段 PyCUDA 代码建立了一个典型的 GPU 图像计算模型：

![](assets/figure-conv2d-02.jpg)

其中，卷积核的半宽、半高不是单纯为了"把尺寸除以二"，而是为了建立一个**以当前输出像素为中心的局部坐标系**。

但当前实现还有五个容易被忽略的问题：

1. 代码实现的是互相关，而 SciPy 验证使用的是真卷积；
2. 循环写法只适用于奇数尺寸卷积核；
3. Grid 使用向下取整，换成 `513×513` 可能漏算边缘；
4. PyCUDA 的 `DeviceAllocation` 与 `GPUArray` `.get()` 接口混用，程序会直接报错；
5. 当前计时包含结果回拷，并不等于纯 Kernel 时间。

理解这五点，才算真正看懂这段代码。

---

## 01 一个线程，到底在计算什么？

![](assets/figure-conv2d-03.jpg)

先看 Kernel 中最重要的两行：

```cpp
int tx = threadIdx.x + blockIdx.x * blockDim.x;
int ty = threadIdx.y + blockIdx.y * blockDim.y;
```

这里：

- `threadIdx`：线程在线程块（block）内部的位置；
- `blockIdx`：线程块在整个线程网格（grid）中的位置；
- `blockDim`：每个线程块包含多少线程。

三者组合起来，得到线程在整张图像上的全局坐标。

假设线程块大小为：

```python
block_size = (16, 16, 1)
```

那么一个 Block 中有 `16 × 16 = 256` 个线程，它们形成一个二维线程矩阵：

![](assets/figure-conv2d-04.jpg)

每个点都是一个 CUDA 线程。

但要注意：

> Block 的 `16×16` 不是输入图像大小，也不是卷积核大小，只是 GPU 调度线程的一种组织方式。多个 Block 拼在一起，才覆盖完整图像。

顺带说明一个常见的误会：图像有多少像素，不代表"就有多少线程同时跑"——程序启动的是**足以覆盖全部输出像素的线程**；Grid 向上取整时还会额外启动少量越界线程，再由 Kernel 的边界判断提前返回。GPU 根据流式多处理器（SM）数量和资源占用，把这些线程块并发或分批调度。线程块（block）是分配到 SM 的基本协作与资源组织单位；进入 SM 后，线程按 32 个线程组成的线程束（warp）被调度执行。NVIDIA 官方把 warp 定义为 CUDA 设备上最小的可执行并行单元（参见 CUDA C++ Best Practices Guide，链接见附录 B）。覆盖图像的其实是"索引公式"，不是 Block 本身。

![](assets/figure-conv2d-05.jpg)

NVIDIA 的 CUDA 编程模型正是通过网格（grid）、线程块（block）和线程（thread）组织并行工作。根据 CUDA Programming Guide 的 execution model，可以把一个 Kernel 的启动理解为"向 GPU 提交一张调度清单，清单里按索引公式给每个线程分配任务"，线程并不负责"抢任务"，索引天然决定了线程与数据的一一对应关系。

---

## 02 为什么一个线程可以代表一个输出像素？

假设某个线程计算得到：

```text
tx = 24
ty = 16
```

它的任务不是处理整张图片，也不是负责卷积核中的某个权重，而是计算输出图像中 `output[16][24]` 这一个像素。

由于二维数组在内存中按行展开，所以：

```cpp
output[ty * input_width + tx]
```

等价于二维写法 `output[ty][tx]`。对于宽度为 `512` 的图像：

```text
output[16][24]
    ↓
output[16 × 512 + 24]
    ↓
output[8216]
```

于是，并行关系就建立起来了：

![](assets/figure-conv2d-06.jpg)

这就是这段程序最核心的并行拆分：

> 输出图像中有多少个像素，逻辑上就准备多少个线程；每个线程独立完成一个输出位置的计算。

不同线程读取的输入区域可能重叠，但它们写入的输出位置不同，因此不需要原子操作，也不需要线程间同步。这里顺便点一下：行号 `ty`、列号 `tx` 是谁，完全由索引公式确定；一个线程之所以"恰好"落在某个像素上，只是因为公式里写了 `blockIdx * blockDim + threadIdx`。

---

## 03 半宽、半高：给卷积窗口建立一套局部坐标

接下来是最容易"看懂代码、却没有建立直觉"的两行：

```cpp
int half_k_width = kernel_width / 2;
int half_k_height = kernel_height / 2;
```

对于一个 `3×3` 卷积核：

```text
kernel_width  = 3
kernel_height = 3
half_k_width  = 3 / 2 = 1
half_k_height = 3 / 2 = 1
```

这里的 `1` 是卷积核的中心到边缘的距离，也就是水平、垂直方向的**邻域半径**。

![](assets/figure-conv2d-07.jpg)

因此下面的循环：

```cpp
for (int i = -half_k_height; i <= half_k_height; i++) {
    for (int j = -half_k_width; j <= half_k_width; j++) {
```

对 `3×3` 卷积核会遍历 `i = -1、0、+1`，`j = -1、0、+1`，一共九个位置。

假设当前线程负责输出像素 `(tx, ty)`，输入坐标写成：

```cpp
int x = tx + j;
int y = ty + i;
```

也就是：

![](assets/figure-conv2d-08.jpg)

所以半宽、半高承担了第一个任务：确定卷积窗口需要以当前输出像素为中心、向四周移动多远。

### 坐标转换：从中心原点回到数组下标

循环里的 `i` 和 `j` 可能是负数（`-1、0、+1`），但数组下标不能是负数。

因此代码又做了一次转换：

```cpp
i + half_k_height
j + half_k_width
```

对于 `3×3` 卷积核：

![](assets/figure-conv2d-09.jpg)

宽度方向同理：

```text
相对偏移 j：         -1    0    +1
卷积核实际列下标：     0    1     2
```

于是：

```cpp
kernel[(i + half_k_height) * kernel_width + (j + half_k_width)]
```

就能把"以中心为原点的相对坐标"转换成"从左上角开始的数组下标"。

这才是半宽、半高最重要的作用：

> 它建立了一座桥，把人更容易理解的中心坐标系，转换为计算机使用的数组坐标系。

换句话说：

```text
半宽、半高 ≠ 只是把卷积核尺寸除以 2
半宽、半高 = 邻域半径 + 坐标原点转换 + 卷积核中心定位
```

---

## 04 Same 模式：为什么跳过越界像素，就等于补零？

当卷积窗口移动到图像中间时，九个输入位置都有效。

但如果当前线程负责左上角（`tx = 0, ty = 0`），`3×3` 窗口的一部分会跑到图像外面：

![](assets/figure-conv2d-10.jpg)

代码通过下面的条件过滤非法坐标：

```cpp
if (x >= 0 && x < input_width && y >= 0 && y < input_height) {
    value += ...;
}
```

图像外的像素没有参与累加，数学效果等价于把它们当成零：

![](assets/figure-conv2d-11.jpg)

这就是零填充（zero padding）。

需要特别区分两个概念：

```text
same 模式：描述输出尺寸（与第一个输入一致）
零填充：描述边界外填什么
```

SciPy 的 `mode='same'` 表示输出尺寸与第一个输入一致；`boundary='fill', fillvalue=0` 才表示使用零填充。因此更准确的说法是：

> 这段 Kernel 通过"越界位置不参与累加"实现与零填充等价的边界行为；同时每个合法输入位置都产生一个输出，所以结果尺寸与输入一致。

---

## 05 第一次反转：这段代码其实不是卷积

现在回到开头的问题。

GPU Kernel 的核心计算是：

```cpp
value += input[y * input_width + x]
       * kernel[(i + half_k_height) * kernel_width + (j + half_k_width)];
```

卷积核被**原样**使用了。比如原始卷积核：

```text
1  2  3
4  5  6
7  8  9
```

当前 GPU 代码仍按这个方向与输入相乘。这种做法叫二维互相关（cross-correlation）：

```text
输入邻域  ×  原样卷积核
```

而数学意义上的二维卷积，计算前需要把卷积核上下、左右各翻转一次，等价于旋转 180°：

```text
翻转后的卷积核：
9  8  7
6  5  4
3  2  1

输入邻域  ×  翻转后的卷积核
```

两者的差别可以概括成一句话：

```text
互相关：直接滑动 Kernel
卷积：先把 Kernel 旋转 180°，再滑动
```

当卷积核恰好中心对称时（例如许多模糊核 `1 2 1 / 2 4 2 / 1 2 1`），翻转前后完全相同，卷积与互相关结果一致。但代码里的卷积核是 `np.random.rand(3, 3)` 生成的随机数，不对称是常态，两个结果自然对不上。

SciPy 正是把这两个操作分成了两个接口：

```python
scipy.signal.convolve2d    # 数学卷积（内部会翻转 Kernel）
scipy.signal.correlate2d   # 互相关（不翻转）
```

![](assets/figure-conv2d-12.jpg)

所以，验证当前 GPU 代码应该用 `correlate2d`：

```python
from scipy.signal import correlate2d

output_cpu = correlate2d(
    input_data,
    kernel,
    mode='same',
    boundary='fill',
    fillvalue=0
)
```

如果坚持使用 `convolve2d`，GPU 端就必须先翻转卷积核。

这里还有一层容易被忽略的等价性：`correlate2d` 的 same 模式用奇数核时，核中心与输出像素的对齐方式（中心对齐）和本文 Kernel 的 `tx + kx - half` 完全一致，这正是验证能通过的前提。若换成偶数核，两个库的"中心锚点"规则可能不一致，数值就会差出索引偏移——这一点在本文 §06 还会遇到。

最后补一句行业语境：深度学习框架里的"卷积层"（如 `torch.nn.Conv2d`）实现的其实是互相关，因为它不翻转核、只做交叉相关。术语如此使用在炼丹社区没问题，但做数值验证时不能只看名字，必须确认两边计算的是不是同一个数学操作。

---

## 06 第二次反转：3×3 能运行，不代表代码支持任意卷积核

原代码使用：

```cpp
int half_k_width = kernel_width / 2;
for (int j = -half_k_width; j <= half_k_width; j++) {
    // ...
}
```

对 `3×3`：`half = 1`，`j = -1、0、+1`，循环三次，没问题。

对 `5×5`：`half = 2`，`j = -2..+2`，循环五次，同样正确。

但换成 `4×4`：

```text
half = 4 / 2 = 2
j = -2、-1、0、+1、+2   ← 循环五次，列只有四列，下标越界
```

原因在于 `[-half, +half]` 是一个天然适合奇数尺寸、有唯一中心点的区间；偶数尺寸的卷积核没有唯一中心像素（4×4 的"中心"是四个位置，不是一个）。

所以教学代码最稳妥的做法是直接限制奇数核：

```python
if kernel_width % 2 == 0 or kernel_height % 2 == 0:
    raise ValueError("当前示例只支持奇数尺寸卷积核")
```

另一种做法是让循环直接遍历卷积核数组，循环次数与核尺寸严格一致：

```cpp
for (int ky = 0; ky < kernel_height; ++ky) {
    for (int kx = 0; kx < kernel_width; ++kx) {
        int y = ty + ky - half_k_height;
        int x = tx + kx - half_k_width;
        // 边界过滤后再相乘累加
    }
}
```

注意：换成遍历写法后用 `ty + ky - half` 重新定位输入坐标，奇数核结果与原来一致；但偶数核的 same 模式仍需明确中心锚点与输出对齐规则。SciPy 官方文档也特别提醒，在偶数长度核的 same 模式下，不同相关接口之间可能存在一个索引偏移。这不是哪一方"错"，而是"中心放哪"没有一致约定——所以教学版本先钉死在奇数核，回避整类歧义。

---

## 07 为什么 512×512 正常，513×513 却可能漏算？

原来的 Grid 计算方式是向下取整：

```python
grid_size = (
    input_width // block_size[0],
    input_height // block_size[1],
    1
)
```

对 `512×512`、Block `16×16`：

```text
512 / 16 = 32   →  32×32 个 Block，刚好覆盖
```

但如果图像宽度变成 `513`：

```text
513 // 16 = 32
32 × 16 = 512   →  最后一个像素没有线程负责
```

宽和高都是 `513` 时，最右侧一列和最下方一行都会整个漏算。

正确做法是向上取整：

```python
grid_size = (
    (input_width + block_size[0] - 1) // block_size[0],
    (input_height + block_size[1] - 1) // block_size[1],
    1
)
```

对 `513`：`(513 + 16 - 1) // 16 = 528 // 16 = 33`。第 33 个 Block 里只有部分线程有效，但 Kernel 内已有边界判断：

```cpp
if (tx < input_width && ty < input_height)
```

超出图像范围的线程不会写入结果。

于是：

![](assets/figure-conv2d-13.jpg)

这是 CUDA 中非常常见的覆盖模式：用"与数据规模解耦的调度清单 + 线程内边界检查"替代"必须整除才安全的侥幸"。与此同源的坑在第 01 篇《GPU 不认识矩阵：它怎样用 1024 个线程算完一个线性层？》（线性层场景）里表现为"1024 条样本恰好整除 block"，这篇则是"图像尺寸恰好整除 16"——本质相同：不要让数据尺寸恰好落在整除巧合上。

## 08 第三次反转：能编译，不代表程序能跑

### 8.1 `DeviceAllocation` 没有 `.get()`

原代码这样分配显存并回读结果：

```python
output_data_gpu = cuda.to_device(output_data)
output_data_gpu.get(output_data)     # ← 这里会抛 AttributeError
```

问题在于：`cuda.to_device()` 返回的是 `pycuda.driver.DeviceAllocation`，它没有 `.get()` 方法——`.get()` 是 `pycuda.gpuarray.GPUArray` 独有。PyCUDA 官方文档明确说明 `to_device()` 会分配设备内存、复制数据，并返回 `DeviceAllocation`。

两套接口二选一，不要混用：

```python
# 方案一：Driver API 风格，回拷用 memcpy_dtoh(dest, src)
cuda.memcpy_dtoh(output_data, output_data_gpu)

# 方案二：GPUArray 风格，回读用 .get()
output_gpu = gpuarray.zeros_like(input_gpu)   # 或 to_gpu/to_gunpy
output_data = output_gpu.get()
```

这个坑与"卷积还是互相关"无关，是 PyCUDA 两层 API 设计差异的必然结果：`cuda.to_device()` 返回 `DeviceAllocation`，回拷要使用 Driver API 的接口（`cuda.memcpy_dtoh`、`cuda.from_device`、`cuda.from_device_like` 等），`.get()` 属于 `GPUArray`，不能直接用于 `DeviceAllocation`。遇到 `AttributeError: 'DeviceAllocation' object has no attribute 'get'` 时，照上面对号入座即可。它的影响比前两个反转更直接——前两个反转只是"算错对象"，这里让程序连一个合法结果都拿不出来。但就算把 `.get()` 修好，代码也未必拿得到可信耗时——这正是 8.2 要讲的事。

### 8.2 修好 `.get()`，原代码依然拿不到任何时间

原代码的计时范围是：

```python
start_time = time.time()
convolution_2d(...)
cuda.Context.synchronize()
output_data_gpu.get(output_data)
end_time = time.time()
```

严格来说，这段原代码**从未得到过任何计时结果**：程序在 `output_data_gpu.get(output_data)` 处抛 `AttributeError`，根本执行不到 `end_time = time.time()`。只有当 `.get()` 被修正为同步的 `cuda.memcpy_dtoh()`、且保持当前计时边界不变时，这个窗口才真正测得东西——它包含：Kernel 启动 + Kernel 执行 + 上下文同步 + 设备到主机（D2H）结果回拷 + 部分 Python 调用开销，既不是纯 Kernel 时间，也不是完整的端到端时间（输入数据早在计时开始之前就已复制到 GPU）。

### 8.3 即使有数字，也未必是纯 Kernel 时间

要得到可信计时，先把测量口径分开：

![](assets/figure-conv2d-14.jpg)

PyCUDA 官方文档把 Event 描述为流（stream）中的时间标记，用于测量两个事件之间的 GPU 工作，例如 Kernel 执行时间；`start.time_till(end)` 返回毫秒值。

顺带提醒：单次 Event 只用来演示 API 用法，不是稳定性能结论——Kernel 首次调用仍受上下文初始化、模块加载与缓存状态影响。正式 benchmark 应先预热、再多次测量取中位数或 P95；附录 A 刻意保持单次计时演示，只为展示正确写法，不产出性能数字。

---

## 09 修正后的 CUDA Kernel

保留当前代码的实际语义（互相关），把前文几个问题一起修掉。关键改动只有四处，其余都与原代码一样：

1. `const float *` 修饰输入与卷积核，向编译器和读者声明"只读"；
2. 输出边界判断改为**早返回**形式（`tx/ty` 越界提前 `return`），与 Grid 向上取整配套，让主计算路径更清晰；
3. 循环改成直接遍历卷积核数组，循环次数与核尺寸严格一致，不再依赖 `[-half, +half]` 的奇数假设；
4. 越界输入位置不参与累加（等价零填充），每个输出位置只写一次（天然契合"一个线程一个像素"，无需原子操作）。

核心片段：

```cpp
// Grid 向上取整后，边缘 Block 里会有多余线程，先过滤
if (tx >= input_width || ty >= input_height) {
    return;
}
// half 仍是奇数核下的邻域半径，用于把核坐标映射回输入坐标
for (int ky = 0; ky < kernel_height; ++ky) {
    for (int kx = 0; kx < kernel_width; ++kx) {
        int x = tx + kx - half_k_width;
        int y = ty + ky - half_k_height;
        if (x >= 0 && x < input_width &&
            y >= 0 && y < input_height) {
            value += input[y * input_width + x]
                   * kernel[ky * kernel_width + kx];
        }
    }
}
output[ty * input_width + tx] = value;
```

这段循环里再无负数下标：`kx` 从 `0` 到 `kernel_width-1`，`kernel[ky * kernel_width + kx]` 永远合法；输入坐标的"越界"交给与零填充等价的 `if` 过滤。凡是没有跑出这条逻辑的"改写"，都只是在重复原代码的隐患。

完整可运行版本见文末附录 A（含断言校验与验证状态）。

---

## 10 真正的卷积，只差一个索引

在本文限定的单通道、奇数核、`stride=1`、相同边界对齐场景里，从互相关改成数学卷积，核心差异收敛到一处：先把核旋转 180°（§05 已讲），再把旋转后的下标放进取核——其余乘加代码一字不改：

```cpp
// 互相关：直接取核
kernel[ky * kernel_width + kx]
// 数学卷积：旋转后取核
kernel[(kernel_height - 1 - ky) * kernel_width + (kernel_width - 1 - kx)]
```

不要因为只差一个索引就低估它。当卷积核不对称时，这个索引决定了边缘方向、梯度方向和最终数值——这正是开头"方向到数值都可能不同"的来源。

---

## 11 怎样验证这份代码：别只比较前 5×5

可信的验证分三步，缺一不可：

1. **选对参考实现**：正文语义是互相关，参考实现用 `correlate2d`（若是卷积语义则用 `convolve2d`），使用相同的 same 模式与零填充；
2. **全量比较，不抽前几行**：肉眼扫 `output_data[:5, :5]` "看起来像"没有任何说服力，用 `np.testing.assert_allclose(output_data, output_cpu, rtol=1e-5, atol=1e-5)` 检查所有元素，超限位置一目了然；
3. **先静态审阅，再上机实测**：本文写作环境（Apple Silicon，无 NVIDIA GPU）没有可运行 CUDA 的环境，附录 A 的代码目前通过静态审阅、尚未实测。发布前须在 NVIDIA GPU 环境补齐真实输出并填妥文末回填表，把"可运行"从承诺变成事实。

不要让我（或任何人）"看起来没问题"说服你：数值一致性只能用断言证明。

---

---

## 12 这段 Kernel 快吗？

它适合教学，但不能因为"跑在 GPU 上"就自称高性能卷积。

当前每个线程都从全局内存读取一个邻域。对 `3×3` 核，从源代码层面看，每个输出像素最多发起 9 次输入元素读取、9 次卷积核元素读取、9 次乘法、约 8 次累加；这些访问最终对应多少次真实显存事务，还会受到缓存命中、合并访问与编译器优化等因素影响。而相邻输出像素的输入窗口高度重叠——输出 A 与输出 B 读取的 3×3 区域可能有 6 个元素相同，朴素实现却让每个线程各自从全局内存重复加载。

合理的优化路线是：

![](assets/figure-conv2d-15.jpg)

几个名词的适用边界：

- 常量内存适合存放较小、只读、且同一 Warp 在相同循环位置读取相同权重的卷积核；
- 共享内存分块（shared memory tiling）让一个 Block 内的线程复用重叠输入区域；
- 可分离卷积把一次二维计算拆成一次横向、一次纵向，复杂度从 O(n²) 降到 O(n)；
- 对复杂卷积、批处理、多通道与深度学习工作负载，优先评估成熟高性能库。

现代 CUDA Programming Guide 强调 Tile Kernel、异步复制与流水线（pipeline）等能力，但**不表示教学 Kernel 必须立即上它们**。同理，异步复制、流水线乃至更新架构上的 TMA（Tensor Memory Accelerator），也不是朴素 `3×3` Kernel 的默认答案；是否值得引入，应由 Nsight Compute 里的内存吞吐、缓存命中率、占用率与 Stall 指标来回答，而不是先上再说。本文只把 TMA 当作进一步演进方向，不暗示当前朴素 Kernel 能直接从它获益。正确的工程顺序永远是：

![](assets/figure-conv2d-16.jpg)

算法没验证、口径没对齐之前谈优化，等于在流沙上盖楼。

## 13 一张表看懂这段代码


| 代码                                           | 实际含义              |
| -------------------------------------------- | ----------------- |
| `tx = blockIdx.x * blockDim.x + threadIdx.x` | 当前线程负责的输出列        |
| `ty = blockIdx.y * blockDim.y + threadIdx.y` | 当前线程负责的输出行        |
| `half_k_width = kernel_width / 2`            | 水平方向的邻域半径         |
| `half_k_height = kernel_height / 2`          | 垂直方向的邻域半径         |
| `x = tx + kx - half_k_width`                 | 当前权重对应的输入列        |
| `y = ty + ky - half_k_height`                | 当前权重对应的输入行        |
| 直接遍历 `ky/kx` 而非 `[-half,+half]`              | 循环次数等于核尺寸，偶数核不再越界 |
| 越界位置不累加                                      | 等价于零填充            |
| 每线程写一个 `output`                              | 无共享写冲突，不需要原子操作    |
| `kx/ky` 从 0 起、不翻转                            | 语义是互相关            |
| `kernel[H-1-ky][W-1-kx]`                     | 旋转 180°，才是数学卷积    |
| Grid 向上取整                                    | 确保任意尺寸都被线程覆盖      |
| CUDA Event 包围 Kernel                         | 测量纯 GPU 时间窗口      |


---

## 14 最后：真正难的不是写出 Kernel，而是说清它算了什么

这段代码最有价值的地方，不是它完成了一次 `3×3` 图像计算，而是把 GPU 编程里几个最基础、也最容易混淆的问题集中到了一起：

![](assets/figure-conv2d-17.jpg)

写出一个能启动的 CUDA Kernel 并不难。真正困难的是，当结果不一致时，你能否准确回答：这个线程读了什么、算了什么、写到了哪里？

半宽、半高只是两次整数除法，但它们背后连接着三套坐标：

![](assets/figure-conv2d-18.jpg)

把这三套坐标真正对齐，二维卷积就不再是一段需要背诵的代码，而是一件非常具体的事：

> GPU 上成千上万个线程，各自站到一个输出像素上，把周围的一小块输入看一遍，完成乘加，然后把答案写回自己的位置。

这就是朴素二维卷积的全部逻辑，也是从"会调用 CUDA"走向"真正理解 CUDA"的第一步。

---

## 延伸：这一次换个战场

本文与线性层文章共享同一批硬课题：`DeviceAllocation.get()`、CUDA Event 口径、线程按索引一一定位输出，以及"先算对、再测对"。区别只在一维与二维：

![](assets/figure-conv2d-19.jpg)

下一篇文章会走进共享内存与分块（tiling）：当"邻域重叠"成为真正的问题时，索引要让位给数据复用，而本篇的"三套坐标"正是理解分块的钥匙。

留一个开放问题（欢迎评论区讨论）：把附录 A 的验证从 `correlate2d` 换成 `convolve2d` 并保持图像 `513×513`，需要同时改哪一处？为什么只改一行代码是危险的？

---

## 附录 A：修正版完整代码

下面这份代码保留原程序语义（互相关），修正了验证对象、奇数核假设、Grid 覆盖、`.get()` 接口与计时口径，并给每个 Kernel 和关键调用加了中文注释，方便对照正文逐段核对。

```python
"""Kernel 能编译，不代表它在算卷积：拆解 PyCUDA 二维卷积的三个反转（第1章 · GPU 编程基础）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：PyCUDA、NumPy、SciPy。
"""
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from scipy.signal import correlate2d

kernel_code = r"""
__global__ void correlation_2d(
    const float *input,      // 输入图像（只读）
    const float *kernel,     // 卷积核（只读）
    float *output,           // 输出图像（每个线程只写一个像素）
    int input_width,
    int input_height,
    int kernel_width,
    int kernel_height)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid 向上取整后会多出无效线程，直接返回
    if (tx >= input_width || ty >= input_height) {
        return;
    }

    // 教学版约定奇数核，half 即邻域半径
    int half_k_width  = kernel_width / 2;
    int half_k_height = kernel_height / 2;

    float value = 0.0f;

    // 直接遍历卷积核数组：循环次数 = 核尺寸，下标永不越界
    for (int ky = 0; ky < kernel_height; ++ky) {
        for (int kx = 0; kx < kernel_width; ++kx) {
            // 核数组坐标 → 相对坐标 → 输入图像绝对坐标
            int x = tx + kx - half_k_width;
            int y = ty + ky - half_k_height;

            // 越界位置等价于零填充，不参与累加
            if (x >= 0 && x < input_width &&
                y >= 0 && y < input_height) {
                value += input[y * input_width + x]
                       * kernel[ky * kernel_width + kx];
            }
        }
    }

    // 一个线程只写一个输出位置：无写冲突，不需要原子操作
    output[ty * input_width + tx] = value;
}
"""

# 固定随机种子，保证实验可重复
rng = np.random.default_rng(42)

input_width = 513           # 故意避开 16 的整数倍，验证 Grid 向上取整
input_height = 513
kernel_width = 3
kernel_height = 3

if kernel_width % 2 == 0 or kernel_height % 2 == 0:
    raise ValueError("当前示例只支持奇数尺寸卷积核")

input_data = rng.random((input_height, input_width), dtype=np.float32)
kernel = rng.random((kernel_height, kernel_width), dtype=np.float32)
output_data = np.empty_like(input_data)

# 编译并获取 Kernel
mod = SourceModule(kernel_code)
correlation_2d = mod.get_function("correlation_2d")

# Host → Device
input_gpu = cuda.to_device(input_data)
kernel_gpu = cuda.to_device(kernel)
output_gpu = cuda.mem_alloc(output_data.nbytes)

block_size = (16, 16, 1)
grid_size = (
    (input_width + block_size[0] - 1) // block_size[0],
    (input_height + block_size[1] - 1) // block_size[1],
    1,
)

# CUDA Event 计时：只包围 Kernel，不含传输与回拷
start_event = cuda.Event()
end_event = cuda.Event()
start_event.record()
correlation_2d(
    input_gpu,
    kernel_gpu,
    output_gpu,
    np.int32(input_width),
    np.int32(input_height),
    np.int32(kernel_width),
    np.int32(kernel_height),
    block=block_size,
    grid=grid_size,
)
end_event.record()
end_event.synchronize()
kernel_time_ms = start_event.time_till(end_event)

# Device → Host
cuda.memcpy_dtoh(output_data, output_gpu)

# 参考实现：互相关用 correlate2d（不翻转核），same + 零填充
output_cpu = correlate2d(
    input_data, kernel, mode="same", boundary="fill", fillvalue=0
).astype(np.float32)

# 全量断言校验
np.testing.assert_allclose(output_data, output_cpu, rtol=1e-5, atol=1e-5)

# 误差量级展示（发回填时一并记录，让读者看到的不只是 PASS）
max_abs_error = float(np.max(np.abs(output_data - output_cpu)))
mean_abs_error = float(np.mean(np.abs(output_data - output_cpu)))

print(f"Kernel 执行时间：{kernel_time_ms:.3f} ms")
print(f"最大绝对误差：{max_abs_error:.8e}")
print(f"平均绝对误差：{mean_abs_error:.8e}")
print("GPU 结果与 SciPy correlate2d 一致")
print(output_data[:5, :5])
```

这段代码保持单次 Event 演示，只展示 API 用法、不产出性能数字；要拿真实耗时的稳定值，请按正文 §08.3 的方式先预热、再多次测量取中位数。

