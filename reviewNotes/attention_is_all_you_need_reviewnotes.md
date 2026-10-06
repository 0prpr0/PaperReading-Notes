# Attention Is All You Need — 复习笔记

> Paper: **Attention Is All You Need**  
> Authors: Ashish Vaswani et al.  
> Venue: NIPS 2017  
> arXiv: 1706.03762  
> 目标：理解 Transformer 为什么提出、核心结构如何工作、为什么 Self-Attention 有优势，以及论文实验如何验证这些设计。

---

## 1. 一句话总结

Transformer 是一种 **完全基于 Attention 的 Encoder-Decoder 序列模型**。  
它去掉了传统序列模型中的 RNN 和 CNN，主要依靠 **Multi-Head Self-Attention + Feed-Forward Network** 建模序列，并通过 **Positional Encoding** 补充位置信息。

---

# 2. 论文解决了什么问题？

在 Transformer 之前，序列建模和机器翻译主要依赖：

- RNN
- LSTM
- GRU
- CNN-based sequence models

这些模型存在一个关键问题：

## RNN 的问题：顺序计算

RNN 中：

$$
h_t = f(h_{t-1}, x_t)
$$

当前位置依赖前一个位置，因此：

```text
h1 → h2 → h3 → h4 → ...
```

无法在同一个样本内部充分并行。

序列越长，训练越受限制。

---

## CNN 的问题：远距离依赖需要多层传播

CNN 可以并行，但局部卷积一次只能看到附近位置。

两个距离很远的 token 要发生信息交互，需要经过多层卷积。

---

## Self-Attention 的关键优势

Self-Attention 可以让任意两个位置直接建立联系：

```text
token1 ───────────────→ token100
```

因此：

- 更容易并行
- 长距离依赖路径更短
- 在典型机器翻译设置下计算成本有竞争力

---

# 3. Transformer 总体结构

Transformer 仍然采用经典的：

```text
Encoder → Decoder
```

原论文中：

- Encoder：6 层
- Decoder：6 层

即：

$$
N = 6
$$

---

# 4. Encoder

每一层 Encoder 包含两个核心子层：

1. Multi-Head Self-Attention
2. Position-wise Feed-Forward Network

结构：

```text
Input
  ↓
Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Feed-Forward Network
  ↓
Add & Norm
  ↓
Output
```

其中：

$$
d_{\text{model}} = 512
$$

---

# 5. Decoder

每一层 Decoder 包含三个核心子层：

1. Masked Multi-Head Self-Attention
2. Encoder-Decoder Attention
3. Feed-Forward Network

结构：

```text
Decoder Input
    ↓
Masked Self-Attention
    ↓
Add & Norm
    ↓
Encoder-Decoder Attention
    ↓
Add & Norm
    ↓
Feed-Forward Network
    ↓
Add & Norm
```

---

# 6. 为什么 Decoder Self-Attention 需要 Mask？

Encoder 的任务是：

> 理解完整输入序列。

所以 Encoder 可以看到整句话。

Decoder 的任务是：

> 根据之前已经生成的 token，预测下一个 token。

因此在预测位置 $i$ 时，不能提前看到未来位置。

例如：

```text
目标：我 爱 你

预测“爱”时：
可以看到：我
不能看到：你
```

所以 Decoder 使用 causal mask。

不合法位置的 attention score 被设为：

$$
-\infty
$$

经过 softmax 后：

$$
e^{-\infty} = 0
$$

于是未来 token 的 attention 权重变成 0。

---

# 7. Attention：Q、K、V

Attention 可以描述为：

> 输入一个 Query 和一组 Key-Value pairs，输出 Value 的加权和。

---

## Query

表示：

> 当前想寻找什么信息？

---

## Key

表示：

> 用什么特征判断某个位置是否与当前 Query 相关？

---

## Value

表示：

> 最终真正被聚合的信息。

---

# 8. Scaled Dot-Product Attention

Transformer 的核心公式：

$$
\boxed{
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
}
$$

拆成四步：

---

## Step 1：计算 Q 和 K 的匹配分数

$$
QK^T
$$

本质是 Query 和所有 Key 做点积。

---

## Step 2：缩放

$$
\frac{QK^T}{\sqrt{d_k}}
$$

为什么要除以：

$$
\sqrt{d_k}
$$

因为当 $d_k$ 较大时，点积数值可能变大，使 softmax 进入梯度很小的区域。

论文解释：

如果 $q$ 和 $k$ 的各分量独立、均值为 0、方差为 1：

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

其方差为：

$$
d_k
$$

标准差就是：

$$
\sqrt{d_k}
$$

所以除以 $\sqrt{d_k}$ 可以控制数值尺度。

---

## Step 3：Softmax

把匹配分数转换成权重：

$$
softmax(score)
$$

得到一组总和为 1 的 attention weights。

---

## Step 4：加权 Value

$$
AttentionWeights \cdot V
$$

得到新的表示。

---

# 9. Multi-Head Attention

Transformer 不只计算一次 Attention，而是并行计算多个 Head。

公式：

$$
MultiHead(Q,K,V)
=
Concat(head_1,\ldots,head_h)W^O
$$

其中：

$$
head_i
=
Attention(QW_i^Q, KW_i^K, VW_i^V)
$$

每个 Head 都有自己的投影矩阵：

$$
W_i^Q,\quad W_i^K,\quad W_i^V
$$

因此不同 Head 可以学习不同的表示关系。

---

## 原论文配置

$$
d_{\text{model}} = 512
$$

$$
h = 8
$$

每个 Head：

$$
d_k = d_v = 64
$$

因为：

$$
512 / 8 = 64
$$

注意：

**不是简单把 512 维机械切成 8 段。**

而是每个 Head 通过自己的线性投影，从完整表示中得到新的 64 维 Q/K/V。

---

# 10. 三种 Attention

Transformer 中一共有三种主要的 Multi-Head Attention。

| 位置 | Q 来源 | K/V 来源 |
|---|---|---|
| Encoder Self-Attention | Encoder | Encoder |
| Decoder Masked Self-Attention | Decoder | Decoder |
| Encoder-Decoder Attention | Decoder | Encoder |

---

## Encoder Self-Attention

Q、K、V 都来自 Encoder 同一层输入。

作用：

> 让输入序列内部不同位置互相交换信息。

---

## Decoder Masked Self-Attention

Q、K、V 都来自 Decoder。

但必须加 causal mask，禁止看未来。

---

## Encoder-Decoder Attention

Query 来自 Decoder。

Key、Value 来自 Encoder。

作用：

> Decoder 根据当前生成状态，从输入序列中寻找相关信息。

也可理解为 Cross-Attention。

---

# 11. Feed-Forward Network

每个 Encoder / Decoder 层里都有 Position-wise FFN：

$$
FFN(x)
=
max(0, xW_1+b_1)W_2+b_2
$$

原论文：

$$
d_{\text{model}} = 512
$$

$$
d_{ff} = 2048
$$

即：

```text
512
 ↓
Linear
 ↓
2048
 ↓
ReLU
 ↓
Linear
 ↓
512
```

关键：

- 同一层中，不同 token 使用同一个 FFN
- 不同 Transformer 层使用不同 FFN 参数

可以简单理解为：

- Attention：token 之间交换信息
- FFN：每个 token 独立进一步加工

---

# 12. Residual Connection + LayerNorm

每个子层使用：

$$
LayerNorm(x + Sublayer(x))
$$

也就是：

```text
Sublayer
   ↓
Residual Add
   ↓
LayerNorm
```

Figure 1 中写作：

```text
Add & Norm
```

Residual Connection 保留原始信息路径。

---

# 13. Embedding

输入 token 首先转换为向量：

```text
token → embedding vector
```

原论文：

$$
d_{\text{model}} = 512
$$

因此每个 token 的主干表示是 512 维。

---

# 14. Positional Encoding（PE）

因为 Transformer 没有 RNN，也没有 CNN，模型本身缺少天然的顺序信息。

所以输入 Transformer 前：

$$
\boxed{
Input = Embedding + PositionalEncoding
}
$$

PE 和 Embedding 的维度相同：

$$
d_{\text{model}} = 512
$$

这样才能直接相加。

---

## Sinusoidal Positional Encoding

论文使用：

$$
PE(pos,2i)
=
\sin
\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right)
$$

$$
PE(pos,2i+1)
=
\cos
\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right)
$$

其中：

- $pos$：token 的位置
- $i$：PE 向量的维度索引
- 偶数维：sin
- 奇数维：cos

---

## 为什么用 sin / cos？

论文给出的动机：

对于固定偏移 $k$：

$$
PE_{pos+k}
$$

可以表示成：

$$
PE_{pos}
$$

的线性函数。

作者认为这有利于模型学习相对位置关系。

---

## Learned PE vs Sinusoidal PE

论文也测试了 learned positional embeddings。

结果：

> 两者效果接近。

作者最终采用 sinusoidal PE，因为认为它可能更有利于外推到训练时未见过的更长序列。

---

# 15. Transformer 中几个维度的区别

| 符号 | 原论文 Base | 含义 |
|---|---:|---|
| $d_{\text{model}}$ | 512 | token 主干表示维度 |
| $h$ | 8 | Attention Head 数 |
| $d_k$ | 64 | 单个 Head 的 Q/K 维度 |
| $d_v$ | 64 | 单个 Head 的 V 维度 |
| $d_{ff}$ | 2048 | FFN 中间层维度 |
| PE 维度 | 512 | 与 $d_{\text{model}}$ 相同 |

一个非常好记的数据流：

$$
512
\rightarrow
8 \times 64
\rightarrow
512
$$

---

# 16. 为什么 Self-Attention？

论文从三个角度比较：

1. Complexity per Layer
2. Sequential Operations
3. Maximum Path Length

---

## Table 1

| Layer Type | Complexity | Sequential Operations | Maximum Path Length |
|---|---:|---:|---:|
| Self-Attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(knd^2)$ | $O(1)$ | $O(\log_k n)$ |
| Restricted Self-Attention | $O(rnd)$ | $O(1)$ | $O(n/r)$ |

---

# 17. Self-Attention 为什么是 $O(n^2d)$？

设：

$$
Q,K \in \mathbb{R}^{n\times d}
$$

那么：

$$
QK^T
$$

维度：

$$
(n\times d)(d\times n)
=
n\times n
$$

因为每个 token 都要和所有 token 做匹配。

一共有：

$$
n^2
$$

组 token pair。

每次点积涉及 $d$ 个维度：

$$
\boxed{
O(n^2d)
}
$$

---

# 18. RNN 为什么是 $O(nd^2)$？

RNN 每个时间步大致有：

$$
d \rightarrow d
$$

的矩阵变换。

对应：

$$
d\times d
$$

权重矩阵。

单步：

$$
O(d^2)
$$

共有 $n$ 个时间步：

$$
\boxed{
O(nd^2)
}
$$

这里的 $d^2$ 来自 hidden state 的线性变换，不是卷积。

---

# 19. CNN 为什么是 $O(knd^2)$？

每个位置：

- 看 $k$ 个邻域位置
- 每个位置有 $d$ 个输入通道
- 输出也是 $d$ 个通道

因此每个位置约：

$$
k d^2
$$

整个序列：

$$
\boxed{
O(knd^2)
}
$$

---

# 20. Self-Attention 有 $n^2$，为什么仍然有优势？

Self-Attention：

$$
O(n^2d)
$$

RNN：

$$
O(nd^2)
$$

两者相比：

$$
\frac{n^2d}{nd^2}
=
\frac{n}{d}
$$

当：

$$
n < d
$$

时：

$$
\frac nd < 1
$$

因此在论文讨论的典型机器翻译设置中，Self-Attention 可以比 recurrent layer 更高效。

---

# 21. Sequential Operations

这是 Transformer 的重要优势。

RNN：

$$
O(n)
$$

因为：

```text
h1 → h2 → h3 → ... → hn
```

必须逐步执行。

Self-Attention：

$$
O(1)
$$

因为所有位置之间的关系可以通过大型矩阵运算并行计算。

注意：

> $O(1)$ 不是说总计算量是常数，而是指必须顺序执行的操作层数是常数级。

---

# 22. Maximum Path Length

RNN：

$$
O(n)
$$

远距离信息需要一路传播。

CNN：

$$
O(\log_k n)
$$

Self-Attention：

$$
O(1)
$$

任意两个 token 可以直接建立联系。

这有利于学习 long-range dependencies。

---

# 23. Restricted Self-Attention

Self-Attention 的缺点：

$$
O(n^2)
$$

序列很长时成本会变大。

论文提出，可以限制每个位置只看邻域大小 $r$。

复杂度变为：

$$
O(rnd)
$$

但最大路径长度变为：

$$
O(n/r)
$$

也就是：

> 计算更便宜，但远距离信息不再一步直达。

---

# 24. Training

## Dataset

### WMT 2014 English-German

约：

$$
4.5M
$$

sentence pairs。

使用 Byte-Pair Encoding，词表约：

$$
37000
$$

tokens。

### WMT 2014 English-French

约：

$$
36M
$$

sentence pairs。

词表约：

$$
32000
$$

word-pieces。

---

# 25. Batching

论文把长度相近的句子放到同一 batch。

目的是：

> 减少 padding 和无效计算。

每个 batch 大约包含：

- 25000 source tokens
- 25000 target tokens

---

# 26. Transformer Base vs Big

两者结构思想相同，只是模型规模不同。

| 参数 | Base | Big |
|---|---:|---:|
| $N$ | 6 | 6 |
| $d_{\text{model}}$ | 512 | 1024 |
| $d_{ff}$ | 2048 | 4096 |
| Heads | 8 | 16 |
| Dropout | 0.1 | 0.3 |
| Training Steps | 100K | 300K |
| Parameters | ~65M | ~213M |

---

# 27. Hardware

使用：

```text
8 × NVIDIA P100 GPU
```

Base：

- 每 step 约 0.4 s
- 100K steps
- 约 12 小时

Big：

- 每 step 约 1.0 s
- 300K steps
- 约 3.5 天

---

# 28. Optimizer

使用 Adam：

$$
\beta_1 = 0.9
$$

$$
\beta_2 = 0.98
$$

$$
\epsilon = 10^{-9}
$$

---

# 29. Learning Rate

学习率决定：

> 每次参数更新沿梯度方向走多大一步。

简化：

$$
w_{new}
=
w_{old}
-
\eta \frac{\partial L}{\partial w}
$$

其中：

$$
\eta
$$

就是 learning rate。

---

## Transformer Learning Rate Schedule

论文：

$$
lrate
=
d_{\text{model}}^{-0.5}
\cdot
\min(
step^{-0.5},
step\cdot warmup^{-1.5}
)
$$

其中：

$$
warmup\_steps = 4000
$$

整体趋势：

```text
小学习率
    ↓
逐渐增加
    ↓
达到峰值
    ↓
逐渐下降
```

即：

> Warmup → Decay

---

# 30. Regularization

论文主要用了：

- Dropout
- Label Smoothing

---

## Dropout

Base：

$$
P_{drop}=0.1
$$

训练时随机丢弃部分激活，减轻过拟合。

---

## Label Smoothing

论文使用：

$$
\epsilon_{ls}=0.1
$$

不让目标分布过度极端。

作者观察：

- perplexity 变差
- accuracy / BLEU 反而提高

说明：

> 单个指标变差，不一定代表模型整体变差。

---

# 31. Experimental Results

## Machine Translation

WMT 2014 English-German：

Transformer Base：

$$
27.3\ BLEU
$$

Transformer Big：

$$
28.4\ BLEU
$$

Big model 超过此前最好结果（包括 ensemble）超过 2 BLEU。

---

# 32. Model Variations

论文通过改变 Transformer 的配置来分析各组件的重要性。

---

## Attention Head 数量

结论：

- single head 更差
- 适量 multi-head 更好
- head 太多也会下降

因此：

$$
\boxed{
更多Head \neq 一定更好
}
$$

---

## $d_k$

减小 attention key size：

$$
d_k
$$

会损害模型质量。

说明 Q/K compatibility 计算需要足够的表示能力。

---

## Model Size

更大的模型通常表现更好。

这也是 Transformer Big 优于 Base 的原因之一。

---

## Dropout

Dropout 对防止 overfitting 很重要。

---

## Positional Encoding

Sinusoidal PE 与 learned positional embeddings 表现接近。

---

# 33. Generalization：English Constituency Parsing

作者还把 Transformer 用在：

> English Constituency Parsing

目的：

> 验证 Transformer 是否只适用于机器翻译。

该任务特点：

- 输出有强结构约束
- 输出可能比输入更长
- 小数据条件具有挑战

Transformer 仍然获得了很强的表现。

说明：

> Transformer 可以推广到其他 sequence transduction task。

---

# 34. Attention Visualization

论文附录可视化了不同 Attention Head。

作者观察到：

- 某些 Head 会捕获 long-distance dependencies
- 某些 Head 看起来与 anaphora resolution 有关
- 不同 Head 会表现出不同的结构模式

注意：

论文并不是严格证明：

> 某个 Head 就等于某个固定语言学功能。

而是观察到：

> 不同 Attention Head 出现了具有句法 / 语义结构特征的注意力模式。

---

# 35. 整篇论文的逻辑主线

```text
RNN 顺序计算
       ↓
难以并行
       ↓
CNN 虽然可并行
但远距离依赖需要多层
       ↓
Self-Attention
可以直接连接任意位置
       ↓
彻底去掉 RNN / CNN
       ↓
Transformer
       ↓
Multi-Head Attention
       +
Feed-Forward Network
       +
Positional Encoding
       ↓
更强并行能力
+
更短依赖路径
       ↓
机器翻译实验达到 SOTA
```

---

# 36. 论文核心创新点

## 1. 完全基于 Attention 的序列模型

核心创新不是：

> 发明 Attention。

而是：

> 用 Attention 替代传统序列建模中的 recurrence 和 convolution，构建完整的 Encoder-Decoder 模型。

---

## 2. Multi-Head Attention

让模型能够从不同表示子空间、不同位置学习多种关系。

---

## 3. Positional Encoding

解决去掉 recurrence 后，模型缺乏天然词序信息的问题。

---

## 4. 强并行能力

Self-Attention 不需要像 RNN 一样按 token 顺序执行。

---

## 5. 更短的长距离依赖路径

任意 token 可以通过 Self-Attention 直接建立联系。

---

# 37. 阅读后应能回答的问题

### Q1：Transformer 为什么要去掉 RNN？

RNN 具有固有的顺序计算限制，难以充分并行。

---

### Q2：Transformer 为什么需要 Positional Encoding？

因为没有 RNN / CNN 后，模型缺少天然的顺序信息。

---

### Q3：Self-Attention 的核心公式是什么？

$$
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

---

### Q4：为什么除以 $\sqrt{d_k}$？

控制点积数值尺度，避免 softmax 在 $d_k$ 较大时进入梯度很小的区域。

---

### Q5：为什么需要 Multi-Head？

让模型可以在不同表示子空间、不同位置上同时学习多种关系。

---

### Q6：Decoder 为什么需要 Mask？

为了保持 autoregressive property，预测当前位置时不能看到未来 token。

---

### Q7：为什么 Encoder 不需要 causal mask？

因为 Encoder 的输入序列本身就是完整已知的。

---

### Q8：为什么 Self-Attention 是 $O(n^2d)$？

因为 $n$ 个 token 两两计算 attention，共 $n^2$ 组关系，每次涉及 $d$ 维点积。

---

### Q9：为什么 RNN 是 $O(nd^2)$？

每个时间步需要大约 $d\times d$ 的矩阵变换，共 $n$ 步。

---

### Q10：为什么 CNN 是 $O(knd^2)$？

每个位置查看 $k$ 个邻域位置，并进行 $d\rightarrow d$ 的通道变换。

---

### Q11：Self-Attention 有 $n^2$，为什么仍然有优势？

论文讨论的典型设置中常有：

$$
n<d
$$

而且 Self-Attention：

- 更容易并行
- Maximum Path Length 更短

所以不能只看 $n^2$。

---

### Q12：Base 和 Big 有什么区别？

架构思想相同。

Big：

- 更宽
- 更多 Head
- 更多参数
- 训练更久

---

# 38. 最后一分钟复习版

如果只剩一分钟，记住：

```text
Transformer
=
Attention Only Architecture

Encoder:
Self-Attention
+
FFN

Decoder:
Masked Self-Attention
+
Cross-Attention
+
FFN

核心公式：

Attention(Q,K,V)
=
softmax(QKᵀ / √dk)V

Base:
dmodel = 512
heads = 8
dff = 2048

PE:
Embedding + Positional Encoding

Self-Attention:
复杂度 O(n²d)
Sequential Operations O(1)
Maximum Path Length O(1)

RNN:
复杂度 O(nd²)
Sequential Operations O(n)
Maximum Path Length O(n)

论文核心：
去掉 RNN / CNN，
用 Self-Attention 建模序列，
提高并行性并缩短长距离依赖路径。
```

---

# 39. Paper Reading Checklist

- [x] Abstract
- [x] Introduction
- [x] Background
- [x] Model Architecture
- [x] Encoder / Decoder
- [x] Scaled Dot-Product Attention
- [x] Multi-Head Attention
- [x] Feed-Forward Network
- [x] Embedding
- [x] Positional Encoding
- [x] Why Self-Attention
- [x] Training
- [x] Experimental Results
- [x] Model Variations
- [x] Generalization
- [x] Conclusion
- [x] Attention Visualizations

---

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.  
**Attention Is All You Need.**  
NIPS 2017.
