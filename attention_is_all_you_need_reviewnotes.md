# Attention Is All You Need：论文笔记

> Vaswani et al., *Attention Is All You Need*, NeurIPS 2017  
> [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## TL;DR

Transformer 保留 Encoder–Decoder 框架，但用 **Multi-Head Attention + Feed-Forward Network** 取代 RNN/CNN 作为序列交互的核心，并用 **Positional Encoding** 补充顺序信息。

它的主要优势是训练并行度高、任意两个 token 的信息传递路径短；主要代价是标准 Self-Attention 对序列长度有 $O(n^2)$ 的时间和显存开销。

> “Attention Is All You Need”不表示模型中只有 Attention；完整模型还包含 FFN、残差连接、LayerNorm、Embedding 和位置编码。

## 1. 问题与动机

Transformer 之前，机器翻译主要依赖 RNN/LSTM/GRU 或 CNN Encoder–Decoder。

- **RNN**：$h_t=f(h_{t-1},x_t)$，当前位置依赖前一状态，同一句话中的时间步难以并行。
- **CNN**：可以并行，但局部卷积需要堆叠多层才能连接远距离位置。
- **Self-Attention**：任意两个位置可以在一层内直接交互，并可用矩阵运算同时处理所有位置。

论文的核心问题是：**能否完全去掉 recurrence 和 convolution，只靠 Attention 构建高质量的序列转换模型？**

## 2. 总体架构

Base 模型采用 6 层 Encoder 和 6 层 Decoder，主干维度为 $d_{\text{model}}=512$。

~~~text
Encoder
Input → Embedding + PE
      → [Self-Attention → Add & Norm → FFN → Add & Norm] × 6
      → Encoder Output

Decoder
Shifted Output → Embedding + PE
               → [Masked Self-Attention → Add & Norm
                  → Cross-Attention       → Add & Norm
                  → FFN                   → Add & Norm] × 6
               → Linear → Softmax → Next-token probabilities
~~~

| | Encoder | Decoder |
|---|---|---|
| Self-Attention 范围 | 整个输入序列 | 当前及之前的目标位置 |
| Causal mask | 不需要 | 需要 |
| Cross-Attention | 无 | 读取 Encoder 输出 |
| 作用 | 理解输入 | 根据输入和已生成前文预测下一个 token |

### Residual、LayerNorm 与 FFN

原论文对子层采用 post-norm：

$$
\operatorname{LayerNorm}\bigl(x+\operatorname{Sublayer}(x)\bigr)
$$

Position-wise FFN 对每个位置独立使用同一组参数：

$$
\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

Base 模型的维度变化为：

$$
512 \rightarrow 2048 \rightarrow 512
$$

Attention 负责 **token 之间交换信息**，FFN 负责 **对每个 token 的表示进行非线性加工**。输出回到 512 维，便于残差相加和继续堆叠。

## 3. Scaled Dot-Product Attention

### Q、K、V

- **Query**：当前位置想寻找什么信息。
- **Key**：每个位置提供什么特征供 Query 匹配。
- **Value**：匹配后真正被加权汇总的信息。

核心公式：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

若有 $n_q$ 个 Query、$n_k$ 个 Key–Value 对：

$$
Q\in\mathbb{R}^{n_q\times d_k},\qquad
K\in\mathbb{R}^{n_k\times d_k},\qquad
V\in\mathbb{R}^{n_k\times d_v}
$$

计算过程：

1. $QK^\top$：得到每个 Query 与每个 Key 的匹配分数，形状为 $n_q\times n_k$。
2. 除以 $\sqrt{d_k}$：控制点积分数的尺度。
3. Softmax：把每行分数转成和为 1 的注意力权重。
4. 乘以 $V$：按权重汇总信息，输出形状为 $n_q\times d_v$。

### 为什么除以 $\sqrt{d_k}$？

若 $q_i,k_i$ 相互独立、均值为 0、方差为 1：

$$
q\cdot k=\sum_{i=1}^{d_k}q_i k_i,\qquad
\operatorname{Var}(q\cdot k)=d_k
$$

点积的标准差约为 $\sqrt{d_k}$。维度增大时，分数绝对值容易变大，使 Softmax 过早接近 one-hot，梯度随之变小。除以 $\sqrt{d_k}$ 可以把分数拉回较稳定的尺度。

## 4. Multi-Head Attention

$$
\operatorname{MultiHead}(Q,K,V)
=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O
$$

$$
\operatorname{head}_i
=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

Base 模型中：

$$
d_{\text{model}}=512,\qquad h=8,\qquad d_k=d_v=64
$$

每个 head 使用独立投影矩阵，从完整表示中学习不同的子空间；8 个 $n\times64$ 输出拼接为 $n\times512$，再经 $W^O$ 融合。

多头提供了从不同位置和表示子空间汇总信息的机会，但**不能据此断言每个 head 必然对应一种固定语法关系**。

### Transformer 中的三种 Attention

| 类型 | Query 来源 | Key / Value 来源 | 可见范围 |
|---|---|---|---|
| Encoder Self-Attention | Encoder 前一层 | Encoder 前一层 | 整个输入 |
| Decoder Self-Attention | Decoder 前一层 | Decoder 前一层 | 当前及之前的位置 |
| Cross-Attention | Decoder | Encoder 输出 | 整个输入 |

Decoder 使用 causal mask。对于长度为 4 的序列：

$$
M=
\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}
$$

$$
\operatorname{MaskedAttention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

未来位置加上 $-\infty$ 后，Softmax 权重变为 0，从而防止训练时偷看未来目标词。Encoder 不需要 causal mask，因为完整输入在编码前已经给出。

## 5. Embedding 与 Positional Encoding

Embedding 把 token 转为 $d_{\text{model}}$ 维向量；Decoder 顶部的 Linear + Softmax 再把隐藏向量转换成整个词表上的下一个 token 概率。论文共享输入/输出 Embedding 与 pre-softmax 线性层的权重。

Self-Attention 没有内置词序，因此模型输入为：

$$
\text{Input}=\text{Token Embedding}+\text{Positional Encoding}
$$

原论文使用固定的正弦位置编码：

$$
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

- $pos$ 是 token 位置，$i$ 是位置向量的维度索引。
- 不同维度使用不同频率，提供多尺度位置信息。
- sin/cos 的移位关系使固定偏移 $pos+k$ 可由 $pos$ 的编码线性表示，可能有助于学习相对位置。

论文中 learned positional embedding 与 sin/cos 效果近似。作者选择后者，是因为它可能更容易外推到训练时未见过的长度，而不是因为实验已经证明它明显更优。

## 6. 为什么选择 Self-Attention？

设序列长度为 $n$、表示维度为 $d$、卷积核宽度为 $k$、局部窗口为 $r$：

| Layer | Complexity per Layer | Sequential Operations | Maximum Path Length |
|---|---:|---:|---:|
| Self-Attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(knd^2)$ | $O(1)$ | $O(\log_k n)$ |
| Restricted Self-Attention | $O(rnd)$ | $O(1)$ | $O(n/r)$ |

关键结论：

- Self-Attention 的所有位置可用矩阵运算并行处理；这里的 $O(1)$ 指顺序阶段数，不是总计算量。
- 任意两个位置可在一层内直接连接，Maximum Path Length 为 $O(1)$。
- 当典型翻译任务中 $n<d$ 时，$O(n^2d)$ 可能低于 RNN 的 $O(nd^2)$。
- 长序列仍是弱点：序列长度翻倍时，注意力矩阵约增至四倍。
- Restricted Attention 用较低成本换取更长的信息传递路径。

## 7. 训练与实验

### 训练配置

| 项目 | Base | Big |
|---|---:|---:|
| Layers | 6 | 6 |
| $d_{\text{model}}$ | 512 | 1024 |
| $d_{ff}$ | 2048 | 4096 |
| Heads | 8 | 16 |
| Dropout | 0.1 | 0.3 |
| Steps | 100K | 300K |
| Parameters | 约 65M | 约 213M |
| 8×P100 训练时间 | 约 12 小时 | 约 3.5 天 |

优化器为 Adam：$\beta_1=0.9$、$\beta_2=0.98$、$\epsilon=10^{-9}$。学习率先 warmup 4,000 步，再衰减：

$$
\text{lrate}=d_{\text{model}}^{-1/2}
\min\left(\text{step}^{-1/2},\ \text{step}\cdot\text{warmup}^{-3/2}\right)
$$

其他关键设置包括 BPE/WordPiece、按相近长度组 batch、Dropout 和 $\epsilon_{ls}=0.1$ 的 Label Smoothing。

### 主要结果

| 任务 | Base | Big | 结论 |
|---|---:|---:|---|
| WMT14 英德 | 27.3 BLEU | **28.4 BLEU** | 超过此前结果，包括集成模型 |
| WMT14 英法 | 38.1 BLEU | **41.8 BLEU** | 达到当时单模型 SOTA |

> Abstract 和 Table 2 报告英法 41.8 BLEU，而 Section 6.1 正文写作 41.0，原文存在数字不一致。

### Ablation 结论

- 单 head 弱于合理数量的 multi-head，但 head 过多也会下降。
- 减小 $d_k$ 会损害效果，说明 Q/K 匹配需要足够容量。
- 更大的模型通常效果更好，同时增加参数与成本。
- Dropout 和 Label Smoothing 有助于泛化。
- Learned PE 与 sinusoidal PE 的结果近似。
- 英语成分句法分析初步说明 Transformer 可用于翻译以外的任务。

实验有力支持“无 RNN/CNN 也能获得高质量翻译”和“训练效率有竞争力”。不过，训练成本来自跨论文估算，并非统一硬件和实现下的受控比较；论文也没有充分证明自回归推理一定更快。

## 8. 创新、局限与影响

### 核心创新

1. 构建首个完全以 Self-Attention 为主要序列交互机制的 Encoder–Decoder。
2. 使用 Multi-Head Attention 在多个表示子空间中并行建模关系。
3. 通过位置编码补回去掉 recurrence 后缺失的顺序信息。
4. 在翻译任务中同时展示高质量和高训练并行度。

Attention、Encoder–Decoder、残差连接和 LayerNorm 都不是本文发明；真正贡献是新的架构组合、尺度化设计和实验验证。

### 局限

- 全局 Self-Attention 的时间和显存成本随序列长度按 $O(n^2)$ 增长。
- 自回归 Decoder 在推理时仍需逐 token 生成。
- 主要实验集中在机器翻译，跨任务证据有限。
- Attention 可视化是定性案例，不能证明每个 head 都有稳定、可解释的语言学功能。

### 影响

Transformer 把序列建模从“按时间步递归传递状态”转为“通过 Attention 全局交换信息”，更适合并行硬件和大规模预训练，后来成为 BERT、GPT、T5 以及大量视觉和多模态模型的基础架构。

## 9. 一分钟复习

~~~text
Transformer = Encoder–Decoder without recurrence/convolution

Encoder:
Self-Attention → Add & Norm → FFN → Add & Norm

Decoder:
Masked Self-Attention → Cross-Attention → FFN

Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V

Base:
d_model = 512, heads = 8, d_k = d_v = 64, d_ff = 2048

优势：训练并行度高、远距离路径短
代价：标准 Self-Attention 为 O(n²)，自回归生成仍是顺序的
~~~

## 10. 自测

1. Transformer 为什么要去掉 RNN 式 recurrence？
2. Q、K、V 分别扮演什么角色？
3. 为什么 Attention 分数要除以 $\sqrt{d_k}$？
4. Multi-Head Attention 与单 Head 的主要区别是什么？
5. Decoder 为什么需要 causal mask，而 Encoder 不需要？
6. Positional Encoding 解决了什么问题？
7. 为什么 Sequential Operations 是 $O(1)$，但总复杂度不是 $O(1)$？
8. 哪些实验真正支持了论文的核心 claim？

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need.** *Advances in Neural Information Processing Systems 30*.
