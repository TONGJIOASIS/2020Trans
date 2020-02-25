# 词嵌入基础

我们在[“循环神经网络的从零开始实现”](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html)一节中使用 one-hot 向量表示单词，虽然它们构造起来很容易，但通常并不是一个好选择。一个主要的原因是，one-hot 词向量无法准确表达不同词之间的相似度，如我们常常使用的余弦相似度。

Word2Vec 词嵌入工具的提出正是为了解决上面这个问题，它将每个词表示成一个定长的向量，并通过在语料库上的预训练使得这些向量能较好地表达不同词之间的相似和类比关系，以引入一定的语义信息。基于两种概率模型的假设，我们可以定义两种 Word2Vec 模型：
1. [Skip-Gram 跳字模型](https://zh.d2l.ai/chapter_natural-language-processing/word2vec.html#%E8%B7%B3%E5%AD%97%E6%A8%A1%E5%9E%8B)：假设背景词由中心词生成，即建模 $P(w_o\mid w_c)$，其中 $w_c$ 为中心词，$w_o$ 为任一背景词；

![Image Name](https://cdn.kesci.com/upload/image/q5mjsq84o9.png?imageView2/0/w/960/h/960)

2. [CBOW (continuous bag-of-words) 连续词袋模型](https://zh.d2l.ai/chapter_natural-language-processing/word2vec.html#%E8%BF%9E%E7%BB%AD%E8%AF%8D%E8%A2%8B%E6%A8%A1%E5%9E%8B)：假设中心词由背景词生成，即建模 $P(w_c\mid \mathcal{W}_o)$，其中 $\mathcal{W}_o$ 为背景词的集合。

![Image Name](https://cdn.kesci.com/upload/image/q5mjt4r02n.png?imageView2/0/w/960/h/960)


1. PTB 数据集
2. Skip-Gram 跳字模型
3. 负采样近似
4. 训练模型


### 二次采样

文本数据中一般会出现一些高频词，如英文中的“the”“a”和“in”。通常来说，在一个背景窗口中，一个词（如“chip”）和较低频词（如“microprocessor”）同时出现比和较高频词（如“the”）同时出现对训练词嵌入模型更有益。因此，训练词嵌入模型时可以对词进行二次采样。 具体来说，数据集中每个被索引词 $w_i$ 将有一定概率被丢弃，该丢弃概率为


$$

P(w_i)=\max(1-\sqrt{\frac{t}{f(w_i)}},0)

$$


其中 $f(w_i)$ 是数据集中词 $w_i$ 的个数与总词数之比，常数 $t$ 是一个超参数（实验中设为 $10^{−4}$）。可见，只有当 $f(w_i)>t$ 时，我们才有可能在二次采样中丢弃词 $w_i$，并且越高频的词被丢弃的概率越大。

## Skip-Gram 跳字模型

在跳字模型中，每个词被表示成两个 $d$ 维向量，用来计算条件概率。假设这个词在词典中索引为 $i$ ，当它为中心词时向量表示为 $\boldsymbol{v}_i\in\mathbb{R}^d$，而为背景词时向量表示为 $\boldsymbol{u}_i\in\mathbb{R}^d$ 。设中心词 $w_c$ 在词典中索引为 $c$，背景词 $w_o$ 在词典中索引为 $o$，我们假设给定中心词生成背景词的条件概率满足下式：


$$

P(w_o\mid w_c)=\frac{\exp(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}{\sum_{i\in\mathcal{V}}\exp(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}

$$
