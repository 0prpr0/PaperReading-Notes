# Attention Is All You Need：论文复习笔记

> **Paper:** *Attention Is All You Need*  
> **Authors:** Ashish Vaswani et al.  
> **Venue:** NIPS 2017  
> **arXiv:** [1706.03762](https://arxiv.org/abs/1706.03762)

## 论文概览

### 这篇论文解决了什么问题？

论文研究的是 **sequence transduction**，尤其是机器翻译：给定一个输入序列，生成另一个输出序列。

作者想解决的核心问题是：

> 能否在不使用 RNN 式 recurrence 和 CNN 式 convolution 的情况下，构建一个效果好、训练快、能够处理长距离依赖的完整 Encoder–Decoder 模型？

### 以前的方法有什么问题？

当时最常用的是 RNN、LSTM、GRU 或 CNN 序列模型。

- **RNN/LSTM/GRU**：第 $t$ 个位置依赖第 $t-1$ 个位置的状态，必须按顺序计算，同一句话内部难以充分并行；远距离信息还需要经过较长的状态传递路径。
- **CNN**：同一层可以并行，但卷积通常只覆盖局部窗口，远距离位置需要经过多层普通卷积或扩张卷积才能联系。
- **已有 Attention**：已经能帮助 Decoder 选择 Encoder 中的相关信息，但通常只是 RNN/CNN 架构的辅助模块，而不是整个模型的主体。

### 使用了什么方法解决？

论文提出 **Transformer**：保留 Encoder–Decoder 总体结构，但让 Attention 成为序列位置之间交换信息的核心机制。

主要组件包括：

- Scaled Dot-Product Attention；
- Multi-Head Attention；
- Encoder Self-Attention；
- Masked Decoder Self-Attention；
- Encoder–Decoder Cross-Attention；
- Position-wise Feed-Forward Network；
- Residual Connection 与 Layer Normalization；
- Positional Encoding。

一句话概括：

> 用 Self-Attention 取代 recurrence/convolution，并用 Positional Encoding 补回顺序信息。

### 结果是什么？

- WMT 2014 英德翻译：Transformer big 达到 **28.4 BLEU**，超过此前结果，包括集成模型。
- WMT 2014 英法翻译：Abstract 和 Table 2 报告 **41.8 BLEU**，达到当时单模型最佳水平。
- Transformer base 在 8 块 P100 GPU 上约训练 12 小时；big 模型约训练 3.5 天。
- 英语成分句法分析实验表明，该架构可以推广到翻译以外的任务。
- Ablation 表明合理数量的多头、足够的模型容量、Dropout 和 Label Smoothing 都有帮助；learned PE 与 sinusoidal PE 的结果接近。

### 未来展望是什么？

论文在 Conclusion 中提出三个方向：

1. 将 Transformer 扩展到图像、音频和视频等非文本模态；
2. 研究局部或受限 Attention，降低长输入上的计算成本；
3. 减少生成过程的顺序性。

这些方向也对应原始 Transformer 的两个主要限制：全局 Self-Attention 的 $O(n^2)$ 成本，以及自回归 Decoder 推理时仍需逐 token 生成。

## 1. Introduction

当时的序列建模主流是 RNN/LSTM/GRU Encoder–Decoder。RNN 的状态更新可写为：

$$
h_t=f(h_{t-1},x_t)
$$

因为 $h_t$ 依赖 $h_{t-1}$，同一个样本中的位置必须按时间步顺序计算。序列越长，顺序链越长；显存限制也会减少可以同时放入 batch 的样本数量。

此前虽然出现了许多优化 RNN 的方法，但没有消除这种根本的顺序依赖。Attention 则允许相距很远的位置直接建立联系，但当时通常仍与 RNN 配合使用。

论文因此提出 Transformer：

- 不使用 recurrence；
- 不使用 convolution 完成主要的序列交互；
- 依靠 Attention 建立输入和输出中的全局依赖；
- 通过更高的并行度缩短训练时间。

Introduction 的论证链是：

```text
RNN/LSTM/GRU 很成功
        ↓
但顺序计算难以并行
        ↓
Attention 可以直接建立远距离联系
        ↓
让 Attention 从辅助模块变成模型主体
```

## 2. Background

Transformer 不是第一个尝试减少顺序计算的模型。

ByteNet、ConvS2S 等 CNN 序列模型已经能够并行处理多个位置，但两个远距离位置之间仍需经过多层卷积：

- ConvS2S 的路径长度随距离近似线性增加；
- ByteNet 使用扩张卷积，将路径缩短到对数级；
- Self-Attention 可以在一层中直接连接任意两个位置。

Self-Attention 本身也不是本文发明的。它此前已经用于阅读理解、摘要、文本蕴含和句子表示学习。本文的关键创新定位是：

> 构建第一个完全依靠 Self-Attention 计算输入、输出表示，而不使用序列对齐 RNN 或卷积的 sequence transduction 模型。

Self-Attention 的输出是多个位置 Value 的加权汇总，可能混合不同类型的信息。Multi-Head Attention 通过多个独立表示子空间缓解这一问题。

## 3. Model Architecture

Transformer 仍然采用 Encoder–Decoder：

```text
输入 token
→ Embedding + Positional Encoding
→ 6 层 Encoder
→ Encoder Output
→ 6 层 Decoder
→ Linear
→ Softmax
→ 下一个 token 的概率
```

### 3.1 Encoder and Decoder Stacks

#### Encoder

Encoder 把输入序列 $(x_1,\ldots,x_n)$ 映射为连续表示 $\mathbf z=(z_1,\ldots,z_n)$。它由 $N=6$ 个结构相同但参数独立的层堆叠而成。

每个 Encoder Layer 有两个子层：

1. Multi-Head Self-Attention；
2. Position-wise Feed-Forward Network。

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

Encoder Self-Attention 中，每个位置都可以查看整个输入序列，因此不需要 causal mask。

#### Decoder

Decoder 同样由 6 层组成，但每层有三个子层：

1. Masked Multi-Head Self-Attention；
2. Encoder–Decoder Attention，也称 Cross-Attention；
3. Position-wise Feed-Forward Network。

```text
Shifted Output
  ↓
Masked Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Encoder–Decoder Attention
  ↓
Add & Norm
  ↓
Feed-Forward Network
  ↓
Add & Norm
```

Decoder 采用自回归生成：预测第 $i$ 个 token 时，只能使用此前已经生成的 token。训练时目标序列向右移动一位：

```text
Decoder 输入：<BOS>  I     love   deep
预测目标：       I    love   deep   learning
```

#### Residual Connection 与 LayerNorm

原论文在每个子层外使用残差连接，再进行 LayerNorm，即 post-norm：

$$
\mathrm{LayerNorm}\bigl(x+\mathrm{Sublayer}(x)\bigr)
$$

残差连接保留原始信息路径，并帮助深层网络训练。所有子层输出维度均为 $d_{\text{model}}=512$，因此能够直接与输入相加。

### 3.2 Attention

Attention 将一个 Query 和一组 Key–Value 对映射为输出：

- **Query**：当前位置想寻找什么；
- **Key**：每个位置提供什么特征供 Query 匹配；
- **Value**：匹配后真正传递和汇总的信息。

#### 3.2.1 Scaled Dot-Product Attention

$$
\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

若有 $n_q$ 个 Query、$n_k$ 个 Key–Value 对：

$$
Q\in\mathbb{R}^{n_q\times d_k},\qquad
K\in\mathbb{R}^{n_k\times d_k},\qquad
V\in\mathbb{R}^{n_k\times d_v}
$$

计算分为四步：

1. $QK^\top$：每个 Query 与每个 Key 做点积，得到匹配分数；
2. 除以 $\sqrt{d_k}$：控制分数尺度；
3. Softmax：将每一行转换为和为 1 的权重；
4. 乘 $V$：按权重汇总 Value。

矩阵维度为：

$$
(n_q\times d_k)(d_k\times n_k)=n_q\times n_k
$$

最终输出维度为：

$$
(n_q\times n_k)(n_k\times d_v)=n_q\times d_v
$$

#### 为什么除以 $\sqrt{d_k}$？

若 $q_i,k_i$ 相互独立、均值为 0、方差为 1：

$$
q\cdot k=\sum_{i=1}^{d_k}q_i k_i,\qquad
\mathrm{Var}(q\cdot k)=d_k
$$

点积的标准差约为 $\sqrt{d_k}$。维度增大时，点积分数容易变得很大，使 Softmax 过早接近 one-hot，较小概率位置的梯度也随之变小。

因果关系是：

```text
d_k 增大
→ 点积方差增大
→ Softmax 输入差距变大
→ Softmax 饱和
→ 梯度变小
```

除以 $\sqrt{d_k}$ 可以把点积重新缩放到较稳定的范围。

#### 3.2.2 Multi-Head Attention

$$
\mathrm{MultiHead}(Q,K,V)
=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O
$$

其中：

$$
\mathrm{head}_i
=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

Base 模型使用：

$$
d_{\text{model}}=512,\qquad h=8,\qquad d_k=d_v=64
$$

每个 head 都使用独立的 $W_i^Q,W_i^K,W_i^V$，从完整表示中学习一个 64 维子空间。每个 head 输出 $n\times64$，8 个 head 拼接后得到 $n\times512$，再通过 $W^O$ 融合。

多个 head 使模型能够同时关注不同位置和不同表示子空间。某些 head **可能**学习语法、指代或长距离关系，但论文没有证明每个 head 必然对应一种固定语言学功能。

#### 3.2.3 Attention 的三种用法

| 类型 | Query 来源 | Key / Value 来源 | 可见范围 |
|---|---|---|---|
| Encoder Self-Attention | Encoder 前一层 | Encoder 前一层 | 整个输入 |
| Decoder Self-Attention | Decoder 前一层 | Decoder 前一层 | 当前及之前的位置 |
| Cross-Attention | Decoder | Encoder 输出 | 整个输入 |

Cross-Attention 中，Decoder 的 Query 用来询问当前需要哪些源语言信息，Encoder 输出提供 Key 和 Value。

Decoder Self-Attention 使用 causal mask：

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
\mathrm{MaskedAttention}(Q,K,V)
=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

未来位置加上 $-\infty$ 后，Softmax 权重变为 0。这样训练时即使完整目标句子已经给出，模型也不能偷看未来答案。

### 3.3 Position-wise Feed-Forward Networks

每个 Encoder/Decoder 层都有一个逐位置 FFN：

$$
\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

Base 模型的维度变化为：

$$
512\rightarrow2048\rightarrow512
$$

- 第一层升维，为每个 token 提供更强的非线性表示能力；
- ReLU 引入非线性；
- 第二层降回 512 维，便于残差相加和堆叠下一层；
- 同一层的所有位置共享 FFN 参数，不同 Transformer 层使用不同参数。

可以简单记为：

> Attention 负责 token 之间交流，FFN 负责每个 token 独立加工信息。

### 3.4 Embeddings and Softmax

Embedding 将输入和输出 token 转换为 $d_{\text{model}}$ 维向量。Decoder 顶部使用线性变换和 Softmax，将隐藏表示转换为整个词表上的 next-token 概率。

论文共享以下三处的权重矩阵：

- 输入 Embedding；
- 输出 Embedding；
- pre-softmax 线性变换。

Embedding 会额外乘以 $\sqrt{d_{\text{model}}}$，再与位置编码相加。

### 3.5 Positional Encoding

Self-Attention 本身没有内置的先后顺序。若不加入位置信息，模型难以可靠地区分“猫追老鼠”和“老鼠追猫”。

模型输入为：

$$
\text{Input}=\text{Token Embedding}+\text{Positional Encoding}
$$

原论文使用固定的正弦位置编码：

$$
PE_{(pos,2i)}
=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos,2i+1)}
=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

- $pos$ 是 token 的位置；
- $i$ 是位置向量中的维度索引；
- 偶数维使用 sin，奇数维使用 cos；
- 不同维度采用不同频率，提供多尺度位置信息。

正弦加法公式说明，固定偏移 $k$ 对应的 $PE_{pos+k}$ 可以由 $PE_{pos}$ 通过固定线性变换表示，因此作者推测这种设计有助于学习相对位置。

论文也测试了 learned positional embedding，两者效果几乎相同。作者选择 sinusoidal PE，是因为它可能更容易外推到训练时未见过的长度，而不是因为实验已证明它明显更好。

## 4. Why Self-Attention

论文从三个角度比较 Self-Attention、RNN 和 CNN：

1. **Complexity per Layer**：每层总计算量；
2. **Sequential Operations**：必须顺序执行的最少步骤；
3. **Maximum Path Length**：任意两个位置之间最长的信息传递路径。

设序列长度为 $n$、表示维度为 $d$、卷积核宽度为 $k$、局部窗口为 $r$：

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|---|---:|---:|---:|
| Self-Attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(knd^2)$ | $O(1)$ | $O(\log_k n)$ |
| Restricted Self-Attention | $O(rnd)$ | $O(1)$ | $O(n/r)$ |

### Self-Attention

$QK^\top$ 产生 $n\times n$ 的位置关系矩阵：

$$
(n\times d)(d\times n)=n\times n
$$

因此每层复杂度约为 $O(n^2d)$。所有位置可以组成矩阵同时计算，所以顺序操作为 $O(1)$；任意两点一层直连，所以最大路径也是 $O(1)$。

这里的 $O(1)$ 只表示顺序阶段数，不表示总计算量是常数。

### RNN

每个时间步进行约 $d\times d$ 的状态变换，共 $n$ 步，因此复杂度为 $O(nd^2)$。时间步必须依次执行，顺序操作和最大路径均为 $O(n)$。

Self-Attention 与 RNN 的主要复杂度项之比为：

$$
\frac{n^2d}{nd^2}=\frac{n}{d}
$$

当机器翻译中的序列长度 $n<d$ 时，Self-Attention 的这一部分计算可能低于 RNN。

### CNN 与 Restricted Attention

CNN 能并行，但局部卷积需要多层才能连接远距离位置。Restricted Attention 只查看附近 $r$ 个位置，把复杂度降为 $O(rnd)$，但最大路径增至 $O(n/r)$。

标准 Self-Attention 的主要代价是 $O(n^2)$：

- 序列长度翻倍，注意力矩阵大小约变为四倍；
- 长文本、图像、音频和视频会产生明显的时间与显存压力。

论文还提出 Attention 可能更容易解释，并在附录中展示部分 head 的模式。但这些可视化只是定性案例，不能证明 Attention 权重就是完整因果解释。

## 5. Training

### 数据与 batching

- WMT 2014 英德：约 450 万句对，使用约 37,000 token 的共享 BPE 词表；
- WMT 2014 英法：约 3,600 万句对，使用约 32,000 word-piece 的词表；
- 按近似句长组 batch，减少 padding 和无效计算；
- 每个 batch 约含 25,000 个源 token 和 25,000 个目标 token。

### Base 与 Big

| 项目 | Transformer Base | Transformer Big |
|---|---:|---:|
| Layers | 6 | 6 |
| $d_{\text{model}}$ | 512 | 1024 |
| $d_{ff}$ | 2048 | 4096 |
| Heads | 8 | 16 |
| Dropout | 0.1 | 0.3 |
| Steps | 100K | 300K |
| Parameters | 约 65M | 约 213M |
| 8×P100 训练时间 | 约 12 小时 | 约 3.5 天 |

### Optimizer 与学习率

论文使用 Adam：

$$
\beta_1=0.9,\qquad\beta_2=0.98,\qquad\epsilon=10^{-9}
$$

学习率先 warmup 4,000 步，再按 step 的平方根倒数衰减：

$$
\text{lrate}
=d_{\text{model}}^{-1/2}
\min\left(
\text{step}^{-1/2},
\text{step}\cdot\text{warmup}^{-3/2}
\right)
$$

训练初期参数和梯度不稳定，Warmup 可以避免一开始使用过大的更新；后期逐渐降低学习率，使参数调整更细致。

### Regularization

- **Dropout**：Base 使用 $P_{\text{drop}}=0.1$，作用于子层输出以及 Embedding 与 PE 的和；
- **Label Smoothing**：使用 $\epsilon_{ls}=0.1$，避免模型过度自信；
- Label Smoothing 使 perplexity 变差，但提高 accuracy 和 BLEU，说明单个指标变差不等于整体性能变差。

## 6. Results

### 6.1 Machine Translation

| 模型 | WMT14 英德 BLEU | WMT14 英法 BLEU |
|---|---:|---:|
| ConvS2S | 25.16 | 40.46 |
| GNMT + RL Ensemble | 26.30 | 41.16 |
| ConvS2S Ensemble | 26.36 | 41.29 |
| Transformer Base | 27.3 | 38.1 |
| Transformer Big | **28.4** | **41.8** |

主要结论：

- Transformer base 在英德任务上已经超过表中此前的单模型和集成模型；
- Transformer big 在英德任务上比此前最好结果高出超过 2 BLEU；
- 英法任务取得当时单模型最佳结果；
- Base 模型的估算训练成本明显低于多个对比系统。

需要注意：

- Abstract 和 Table 2 报告英法 41.8 BLEU，Section 6.1 正文写作 41.0，原文存在数字不一致；
- 训练成本是根据不同论文的训练时间、GPU 数量和估算性能比较，并非完全统一条件下的受控实验；
- 论文主要证明训练效率，没有充分证明自回归推理必然更快。

### 6.2 Model Variations

Table 3 的主要结论：

- 单 head 的 BLEU 为 24.9，8/16 个 head 达到 25.8，32 个 head 又降至 25.4；
- Multi-Head 有帮助，但 head 不是越多越好，过多 head 会让单 head 维度过小；
- 减小 $d_k$ 会损害结果，说明 Q/K 匹配需要足够表示容量；
- 增大 $d_{\text{model}}$ 或 $d_{ff}$ 通常提高效果，但参数量和成本也增加；
- 不使用 Dropout 时效果明显下降；
- 不使用 Label Smoothing 时 perplexity 更好，但 BLEU 更差；
- learned PE 为 25.7 BLEU，sinusoidal PE 为 25.8，二者近似。

Ablation 支持多个设计选择，但不能把 Transformer big 的提升全部归功于某一个组件，因为它同时具有更大的模型容量和更长的训练时间。

### 6.3 English Constituency Parsing

作者将四层 Transformer 用于 Penn Treebank 的英语成分句法分析：

- 仅使用约 40,000 个 WSJ 训练句子时达到 91.3 F1；
- 半监督设置下达到 92.7 F1；
- 只进行少量任务特定调参。

这个实验初步说明 Transformer 不只适用于翻译。但它只是一个额外任务，不能证明模型已经普遍适用于所有序列问题。

### 实验对核心 claim 的支持程度

| Claim | 证据 | 判断 |
|---|---|---|
| 不使用 RNN/CNN 也能做好翻译 | 英德、英法 BLEU | 支持较强 |
| 翻译质量达到当时最佳 | 超过多个单模型和集成模型 | 支持较强 |
| 训练效率更高 | 训练时间与估算 FLOPs | 支持，但比较并非完全受控 |
| Multi-Head 优于单 Head | Table 3 消融 | 有直接支持 |
| sinusoidal PE 明显更优 | 与 learned PE 几乎相同 | 不支持明显优势 |
| Transformer 具有通用性 | 句法分析实验 | 初步支持 |
| 每个 head 都学习明确语法功能 | 少量可视化 | 证据不足 |
| 推理一定更快 | 缺少严格推理速度对比 | 未充分证明 |

## 7. Conclusion

论文最终证明：Attention 不必只是 RNN/CNN 的辅助模块，也可以成为完整序列转换模型的核心。

### 核心创新

- 第一个完全以 Attention 为主要序列交互方式的 Encoder–Decoder；
- Scaled Dot-Product Attention 控制高维点积的尺度；
- Multi-Head Attention 允许模型同时从多个表示子空间汇总信息；
- Positional Encoding 为无 recurrence 的模型补充顺序；
- 在机器翻译中同时展示质量与训练并行度优势。

Attention、Encoder–Decoder、残差连接、LayerNorm 和位置编码思想都不是本文单独发明的。真正贡献在于新的架构组合、具体设计和大规模实验验证。

### 原始 Transformer 的局限

- 全局 Self-Attention 的时间和显存成本为 $O(n^2)$；
- 自回归 Decoder 推理时仍需逐 token 生成；
- 主要实验集中在机器翻译，跨任务验证有限；
- Attention 可视化不能直接等同于模型解释。

### 影响

Transformer 将序列建模从“沿时间步递归传递状态”转为“通过 Attention 全局交换信息”，更适合并行硬件和大规模预训练，后来成为 BERT、GPT、T5 以及大量视觉和多模态模型的基础。

### 未来方向

作者提出将模型扩展到非文本模态、研究局部/受限 Attention，并减少生成的顺序性。这些方向分别对应后来的视觉 Transformer、长序列高效 Attention 和非自回归生成等研究路线。

## Appendix：Attention Visualizations

附录 Figure 3–5 展示了部分 Encoder Self-Attention head：

- 有些 head 看起来捕获了长距离依赖；
- 有些 head 似乎与代词指代有关；
- 不同 head 呈现不同的结构模式。

这些图支持“不同 head 可能学习不同关系”的直觉，但属于经过选择的定性案例，不能证明每个 head 都具有稳定、明确的语言学职责，也不能证明 Attention 权重就是预测的完整原因。

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need.** *Advances in Neural Information Processing Systems 30*.
