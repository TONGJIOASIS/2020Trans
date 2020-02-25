# 文本情感分类

文本分类是自然语言处理的一个常见任务，它把一段不定长的文本序列变换为文本的类别。本节关注它的一个子问题：使用文本情感分类来分析文本作者的情绪。这个问题也叫情感分析，并有着广泛的应用。

同搜索近义词和类比词一样，文本分类也属于词嵌入的下游应用。在本节中，我们将应用预训练的词向量和含多个隐藏层的双向循环神经网络与卷积神经网络，来判断一段不定长的文本序列中包含的是正面还是负面的情绪。后续内容将从以下几个方面展开：

1. 文本情感分类数据集
2. 使用循环神经网络进行情感分类
3. 使用卷积神经网络进行情感分类

## 使用循环神经网络

### 双向循环神经网络

在[“双向循环神经网络”](https://zh.d2l.ai/chapter_recurrent-neural-networks/bi-rnn.html)一节中，我们介绍了其模型与前向计算的公式，这里简单回顾一下：

![Image Name](https://cdn.kesci.com/upload/image/q5mnobct47.png?imageView2/0/w/960/h/960)


![Image Name](https://cdn.kesci.com/upload/image/q5mo6okdnp.png?imageView2/0/w/960/h/960)


给定输入序列 $\{\boldsymbol{X}_1,\boldsymbol{X}_2,\dots,\boldsymbol{X}_T\}$，其中 $\boldsymbol{X}_t\in\mathbb{R}^{n\times d}$ 为时间步（批量大小为 $n$，输入维度为 $d$）。在双向循环神经网络的架构中，设时间步 $t$ 上的正向隐藏状态为 $\overrightarrow{\boldsymbol{H}}_{t} \in \mathbb{R}^{n \times h}$ （正向隐藏状态维度为 $h$），反向隐藏状态为 $\overleftarrow{\boldsymbol{H}}_{t} \in \mathbb{R}^{n \times h}$ （反向隐藏状态维度为 $h$）。我们可以分别计算正向隐藏状态和反向隐藏状态：


$$

\begin{aligned}
&\overrightarrow{\boldsymbol{H}}_{t}=\phi\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x h}^{(f)}+\overrightarrow{\boldsymbol{H}}_{t-1} \boldsymbol{W}_{h h}^{(f)}+\boldsymbol{b}_{h}^{(f)}\right)\\
&\overleftarrow{\boldsymbol{H}}_{t}=\phi\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x h}^{(b)}+\overleftarrow{\boldsymbol{H}}_{t+1} \boldsymbol{W}_{h h}^{(b)}+\boldsymbol{b}_{h}^{(b)}\right)
\end{aligned}

$$


其中权重 $\boldsymbol{W}_{x h}^{(f)} \in \mathbb{R}^{d \times h}, \boldsymbol{W}_{h h}^{(f)} \in \mathbb{R}^{h \times h}, \boldsymbol{W}_{x h}^{(b)} \in \mathbb{R}^{d \times h}, \boldsymbol{W}_{h h}^{(b)} \in \mathbb{R}^{h \times h}$ 和偏差 $\boldsymbol{b}_{h}^{(f)} \in \mathbb{R}^{1 \times h}, \boldsymbol{b}_{h}^{(b)} \in \mathbb{R}^{1 \times h}$ 均为模型参数，$\phi$ 为隐藏层激活函数。

然后我们连结两个方向的隐藏状态 $\overrightarrow{\boldsymbol{H}}_{t}$ 和 $\overleftarrow{\boldsymbol{H}}_{t}$ 来得到隐藏状态 $\boldsymbol{H}_{t} \in \mathbb{R}^{n \times 2 h}$，并将其输入到输出层。输出层计算输出 $\boldsymbol{O}_{t} \in \mathbb{R}^{n \times q}$（输出维度为 $q$）：


$$

\boldsymbol{O}_{t}=\boldsymbol{H}_{t} \boldsymbol{W}_{h q}+\boldsymbol{b}_{q}

$$


其中权重 $\boldsymbol{W}_{h q} \in \mathbb{R}^{2 h \times q}$ 和偏差 $\boldsymbol{b}_{q} \in \mathbb{R}^{1 \times q}$ 为输出层的模型参数。不同方向上的隐藏单元维度也可以不同。

利用 [`torch.nn.RNN`](https://pytorch.org/docs/stable/nn.html?highlight=rnn#torch.nn.RNN) 或 [`torch.nn.LSTM`](https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM) 模组，我们可以很方便地实现双向循环神经网络，


## 使用卷积神经网络

### 一维卷积层

在介绍模型前我们先来解释一维卷积层的工作原理。与二维卷积层一样，一维卷积层使用一维的互相关运算。在一维互相关运算中，卷积窗口从输入数组的最左方开始，按从左往右的顺序，依次在输入数组上滑动。当卷积窗口滑动到某一位置时，窗口中的输入子数组与核数组按元素相乘并求和，得到输出数组中相应位置的元素。如图所示，输入是一个宽为 7 的一维数组，核数组的宽为 2。可以看到输出的宽度为 7−2+1=6，且第一个元素是由输入的最左边的宽为 2 的子数组与核数组按元素相乘后再相加得到的：0×1+1×2=2。

![Image Name](https://cdn.kesci.com/upload/image/q5mo8qs7dc.png?imageView2/0/w/960/h/960)

多输入通道的一维互相关运算也与多输入通道的二维互相关运算类似：在每个通道上，将核与相应的输入做一维互相关运算，并将通道之间的结果相加得到输出结果。下图展示了含 3 个输入通道的一维互相关运算，其中阴影部分为第一个输出元素及其计算所使用的输入和核数组元素：0×1+1×2+1×3+2×4+2×(−1)+3×(−3)=2。

![Image Name](https://cdn.kesci.com/upload/image/q5moaawczv.png?imageView2/0/w/960/h/960)

### 时序最大池化层

类似地，我们有一维池化层。TextCNN 中使用的时序最大池化（max-over-time pooling）层实际上对应一维全局最大池化层：假设输入包含多个通道，各通道由不同时间步上的数值组成，各通道的输出即该通道所有时间步中最大的数值。因此，时序最大池化层的输入在各个通道上的时间步数可以不同。

![Image Name](https://cdn.kesci.com/upload/image/q5mobv3kol.png?imageView2/0/w/960/h/960)

*注：自然语言中还有一些其他的池化操作，可参考这篇[博文](https://blog.csdn.net/malefactor/article/details/51078135)。*

为提升计算性能，我们常常将不同长度的时序样本组成一个小批量，并通过在较短序列后附加特殊字符（如0）令批量中各时序样本长度相同。这些人为添加的特殊字符当然是无意义的。由于时序最大池化的主要目的是抓取时序中最重要的特征，它通常能使模型不受人为添加字符的影响。