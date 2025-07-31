# 图像数据集展示函数解析

##### User:

带我讲解一下这个为图像数据集展示图片的函数

```py
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplot(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.istensor(img):
            ax.imshow(img.numpy)
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles)
    return axes
```

并给代码标注类型

##### Assistant:

好的，非常乐意为您深入讲解这个在图像数据处理中非常实用的 `show_images`
函数。作为一个数据科学与深度学习的从业者，这几乎是日常工作中不可或缺的代码片段，用于快速可视化数据集或模型预测结果。

我将首先提供一个经过类型标注和修正的优化版本，然后分步拆解，由浅入深地讲解它的工作原理和设计思想。

### 优化及类型标注后的代码

在原始代码中，存在两个小问题：`d2l.plt.subplot` 应该是 `d2l.plt.subplots`（复数s），以及 `ax.set_title(titles)` 应该使用索引
`titles[i]` 来为每个子图设置对应的标题。同时，为了代码的健壮性和可读性，我们为其添加 Python 的类型提示 (Type Hinting)。

```python
from typing import List, Optional, Tuple, Union
import torch
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def show_images(
        imgs: List[Union[torch.Tensor, np.ndarray]],
        num_rows: int,
        num_cols: int,
        titles: Optional[List[str]] = None,
        scale: float = 1.5
) -> Axes:
    """
    在一个网格中展示一组图像。

    Args:
        imgs (List[Union[torch.Tensor, np.ndarray]]): 包含图像的列表。图像可以是PyTorch张量或NumPy数组。
        num_rows (int): 网格的行数。
        num_cols (int): 网格的列数。
        titles (Optional[List[str]], optional): 每个子图的标题列表。默认为None。
        scale (float, optional): 用于缩放图像尺寸的因子。默认为1.5。
        
    Returns:
        Axes: Matplotlib的坐标轴对象数组。
    """
    # 计算整个图窗(Figure)的尺寸
    figsize: Tuple[float, float] = (num_cols * scale, num_rows * scale)

    # 创建一个包含 num_rows * num_cols 个子图的网格
    # fig 是整个图窗对象, axes 是一个包含所有子图坐标轴的numpy数组
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # 将二维的axes数组(如果是多行多列)展平为一维数组，方便遍历
    axes: np.ndarray = axes.flatten()

    # 使用 zip 将坐标轴对象和图像配对，并用 enumerate 获取索引
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # 判断图像是否为PyTorch张量
        if torch.is_tensor(img):
            # Matplotlib的imshow函数需要NumPy数组作为输入
            # .numpy() 方法将Tensor转换为NumPy array
            ax.imshow(img.numpy())
        else:
            # 如果不是Tensor，假定它已经是Matplotlib兼容的格式 (如NumPy array)
            ax.imshow(img)

        # 为了美观，隐藏每个子图的x和y轴刻度
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        # 如果提供了标题列表
        if titles:
            # 为当前坐标轴(ax)设置对应的标题
            ax.set_title(titles[i])

    return axes
```

---

### 函数功能分步详解

这个函数的核心目标是：**接收一个图像列表，并将它们优雅地排列在一个网格中进行显示**。

#### 1. 函数签名 (Function Signature)

```python
def show_images(
        imgs: List[Union[torch.Tensor, np.ndarray]],
        num_rows: int,
        num_cols: int,
        titles: Optional[List[str]] = None,
        scale: float = 1.5
) -> Axes:
```

* `imgs`: 这是要显示的核心内容——一个图像列表。类型提示 `List[Union[torch.Tensor, np.ndarray]]` 意味着这个列表中的每个元素既可以是
  PyTorch 的 `Tensor`，也可以是 NumPy 的 `ndarray`。这在深度学习中非常常见，因为数据加载器通常输出 `Tensor`，而许多图像处理库使用
  `ndarray`。
* `num_rows` 和 `num_cols`: 这两个整数参数定义了你希望如何排列这些图片，即网格的行数和列数。例如，要显示10张图片，你可以设置为
  `num_rows=2, num_cols=5`。
* `titles`: 这是一个可选参数（`Optional`），默认值为 `None`。如果提供，它应该是一个字符串列表，其长度与 `imgs`
  列表相同，用于给每张图片上方添加一个标题。
* `scale`: 一个浮点数，用作缩放因子来控制每个子图的显示大小，从而影响整个画布的尺寸。`scale` 越大，显示的图片就越大。
* `-> Axes`: 这是返回值类型提示，表示该函数将返回一个 Matplotlib 的 `Axes` 对象（准确说是 `Axes` 对象的数组）。

#### 2. 计算画布尺寸

```python
figsize: Tuple[float, float] = (num_cols * scale, num_rows * scale)
```

* 在 Matplotlib 中，`figsize` 参数接收一个元组 `(width, height)` 来指定整个图窗（Figure）的尺寸，单位通常是英寸。
* 这里的计算逻辑很直观：总宽度 = 列数 × 每列的缩放尺寸，总高度 = 行数 ×
  每行的缩放尺寸。这保证了当你增加行数或列数时，图窗会自动变大以容纳更多的图片，同时保持每个子图的相对大小。

#### 3. 创建子图网格

```python
fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
```

* 这是 Matplotlib 中最核心的函数之一。`plt.subplots()` (注意是复数`s`) 是一个便捷函数，用于一次性创建**一个图窗 (`Figure`)
  ** 和**一组子图 (`Axes`)**。
* `fig`: 是整个画布对象，你可以用它来保存图像 (`fig.savefig(...)`) 或进行其他全局设置。在这个函数里，我们用下划线 `_`
  忽略了它，因为我们主要关心对每个子图的操作。
* `axes`: 这是所有子图坐标轴对象的集合。如果 `num_rows > 1` 或 `num_cols > 1`，`axes` 会是一个二维的 NumPy 数组。例如，对于
  `plt.subplots(2, 3)`，`axes` 的形状是 `(2, 3)`。

#### 4. 展平坐标轴数组

```python
axes: np.ndarray = axes.flatten()
```

* **为什么需要这一步？** `plt.subplots` 返回的 `axes` 是一个二维数组（例如2x5的网格），而我们的 `imgs` 列表是一维的。为了能简单地通过一个
  `for` 循环将图像和子图一一对应，我们需要将二维的 `axes` 数组“压平”成一维数组。
* `flatten()` 方法就是做这个的。例如，一个形状为 `(2, 5)` 的数组会被转换成一个长度为10的一维数组。这样，我们就可以方便地使用
  `zip` 函数了。

#### 5. 遍历和绘制核心逻辑

```python
for i, (ax, img) in enumerate(zip(axes, imgs)):
```

* `zip(axes, imgs)`: `zip` 是一个非常强大的 Python 内置函数。它将 `axes` 列表和 `imgs` 列表像拉链一样合并起来，生成一系列的
  `(ax, img)` 对。循环的每一次迭代，我们都会得到一个坐标轴对象 `ax` 和一张对应的图片 `img`。
* `enumerate(...)`: 这个函数为 `zip` 生成的每一对 `(ax, img)` 添加一个计数器 `i`（从0开始）。我们需要这个索引 `i` 来从
  `titles` 列表中获取正确的标题。

**循环体内部:**

```python
# 1. 处理不同类型的图像数据
if torch.is_tensor(img):
    ax.imshow(img.numpy())
else:
    ax.imshow(img)
```

* `torch.is_tensor(img)`: 检查 `img` 是否为 PyTorch 张量。
* `ax.imshow(img.numpy())`: `ax.imshow()` 是 Matplotlib 中用于绘制图像的函数。然而，它不能直接处理 PyTorch 的 `Tensor`
  。因此，我们必须先调用 `.numpy()` 方法将其转换为 NumPy 数组。
* **注意**: 这里的 `img` 必须是 CPU 上的张量。如果它在 GPU 上，你需要先调用 `img.cpu().numpy()`。
* **图像通道顺序**: `imshow` 期望的图像格式通常是 `(H, W)`（灰度图）或 `(H, W, C)`（彩色图，C是通道数）。PyTorch 中图像张量的常见格式是
  `(C, H, W)`。如果你的张量是这种格式，你需要先转换维度，例如 `ax.imshow(img.permute(1, 2, 0).numpy())`。

```python
# 2. 美化显示
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
```

* 由于我们只是展示图片，坐标轴上的刻度、数字和标签通常是多余的，会干扰视觉效果。这两行代码就是获取 x 轴和 y 轴对象，并将它们设置为不可见。

```python
# 3. 添加标题
if titles:
    ax.set_title(titles[i])
```

* 首先检查 `titles` 列表是否被提供（即不为 `None`）。
* 如果提供了，就使用 `ax.set_title()` 方法为当前的子图 `ax` 设置标题。标题的内容是 `titles` 列表中的第 `i` 个元素，这要归功于
  `enumerate` 提供的索引 `i`。

### 总结

`show_images` 函数是一个设计精良、高度可复用的工具函数。它通过以下几个关键步骤，优雅地解决了批量图像显示的问题：

1. **参数化**：通过参数控制网格布局、尺寸和标题，使其非常灵活。
2. **抽象化**：将 Matplotlib 的复杂设置封装起来，用户只需提供数据和简单的配置即可。
3. **兼容性**：通过 `if torch.is_tensor(img)` 判断，同时支持了深度学习中常见的 PyTorch Tensor 和通用的 NumPy
   数组，增强了函数的适用范围。
4. **代码简洁性**：巧妙地运用了 `zip` 和 `enumerate` 来实现图像和子图的配对遍历，使得核心循环非常简洁易读。

在任何需要可视化图像数据的场景，比如检查数据预处理效果、观察数据增强结果、或展示模型对一批图片的预测输出时，这个函数都非常有用。

