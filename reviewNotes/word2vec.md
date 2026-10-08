# Efficient Estimation of Word Representations in Vector Space

> **Authors:** Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean  
> **Year:** 2013  
> **Keywords:** Word2Vec, CBOW, Skip-gram, Word Embedding, Hierarchical Softmax

---

## 1. Motivation

传统 NLP 中，词通常被当成彼此独立的离散符号，例如词表中的 index 或 one-hot 表示。

这种表示的问题是：

- 无法直接体现词之间的相似性；
- `cat` 和 `dog` 与 `cat` 和 `table` 在 one-hot 空间中没有本质区别；
- 在数据有限时，仅靠扩大训练数据规模很难持续提升效果。

因此，作者希望学习一种 **continuous distributed representation**：

\[
word \rightarrow embedding \in \mathbb{R}^D
\]

使语义或句法相关的词在向量空间中呈现一定规律。

论文的核心目标是：

> **在超大规模语料上，以较低计算成本学习高质量词向量。**

---

## 2. Word Representation

一个词首先对应词表中的 index：

```text
cat -> 127
dog -> 45
```

index 本身只是标识。

概念上可以进一步表示成 one-hot：

\[
cat=[0,\dots,1,\dots,0]
\]

然后通过 Embedding Matrix 得到低维稠密向量：

\[
cat=[0.21,-0.37,0.82,\dots]
\]

实际实现中通常不会真的构造 one-hot，而是直接根据词 ID 查 embedding：

```python
embedding[word_id]
```

训练开始时，各词向量之间通常还没有有意义的语义关系；这些关系是在训练过程中根据上下文逐渐学出来的。

---

## 3. Computational Complexity

论文统一将训练复杂度写为：

\[
O=E\times T\times Q
\]

其中：

- \(E\)：epoch 数；
- \(T\)：训练语料中的词数；
- \(Q\)：每个训练样本的计算复杂度。

作者希望使用非常大的训练语料 \(T\)，因此重点是降低每个样本的计算成本 \(Q\)。

---

## 4. Previous Neural Language Models

### 4.1 Feedforward NNLM

结构：

```text
Input
  ↓
Projection
  ↓
Hidden
  ↓
Output
```

设：

- \(N\)：上下文词数量；
- \(D\)：词向量维度；
- \(H\)：hidden layer 大小；
- \(V\)：词表大小。

复杂度：

\[
Q=ND+NDH+HV
\]

其中：

- \(ND\)：处理 \(N\) 个输入词；
- \(NDH\)：Projection → Hidden；
- \(HV\)：Hidden → Vocabulary。

普通输出层对整个词表计算代价较高，因此可以使用 **Hierarchical Softmax**：

\[
HV \rightarrow H\log_2V
\]

此时主要计算瓶颈来自：

\[
NDH
\]

即复杂的 nonlinear hidden layer。

---

### 4.2 RNNLM

RNN 通过 hidden state 保存历史信息：

\[
h_t=f(x_t,h_{t-1})
\]

复杂度：

\[
Q=H^2+HV
\]

使用 Hierarchical Softmax 后，输出端成本下降，主要计算量来自：

\[
H^2
\]

因此作者发现：

> NNLM 和 RNNLM 的大量计算成本都与 hidden layer 有关。

这直接引出本文的设计思路：

> **如果主要目标只是学习高质量词向量，可以尝试去掉复杂的 nonlinear hidden layer。**

---

# 5. New Log-linear Models

## 5.1 CBOW

CBOW 的方向是：

\[
\boxed{Context \rightarrow Center\ Word}
\]

例如：

```text
The cat [ ? ] sitting on the mat
```

使用周围词：

```text
The, cat, sitting, on
```

预测中心词：

```text
is
```

上下文词的 embedding 会被组合，形成 context representation：

\[
h=\frac{1}{N}\sum_{i=1}^{N}v_i
\]

然后用 \(h\) 预测目标词。

CBOW 不考虑上下文中词的顺序，因此称为 **Bag-of-Words**。

复杂度：

\[
Q=ND+D\log_2V
\]

其中：

- \(ND\)：处理 \(N\) 个上下文词向量；
- \(D\log_2V\)：通过 Hierarchical Softmax 预测目标词。

相比传统 NNLM，CBOW 去掉了昂贵的 nonlinear hidden layer。

---

## 5.2 Skip-gram

Skip-gram 与 CBOW 方向相反：

\[
\boxed{Center\ Word \rightarrow Context}
\]

例如：

```text
I really like drinking hot coffee every morning
```

以：

```text
coffee
```

为中心词，模型尝试预测周围的：

```text
hot
drinking
every
morning
```

其复杂度为：

\[
Q=C(D+D\log_2V)
\]

其中 \(C\) 为最大上下文距离。

窗口越大：

- 可以利用更多上下文；
- 需要进行更多预测；
- 计算成本越高。

论文还使用随机窗口，使离中心词更近的词更频繁地参与训练。

---

## 5.3 CBOW vs Skip-gram

| 项目 | CBOW | Skip-gram |
|---|---|---|
| 输入 | 多个上下文词 | 一个中心词 |
| 输出 | 中心词 | 多个上下文词 |
| 方向 | Context → Word | Word → Context |
| 计算 | 相对更快 | 通常计算更多 |
| 论文实验特点 | syntactic 表现更强 | semantic 表现更强 |

最简单的记忆方式：

\[
\boxed{CBOW:\ Context\rightarrow Word}
\]

\[
\boxed{Skip\text{-}gram:\ Word\rightarrow Context}
\]

---

## 6. Hierarchical Softmax

普通 Softmax 对整个词表进行打分。

假设当前表示为 \(h\)，某个词的输出向量为 \(u_i\)，则：

\[
score(w_i)=h^Tu_i
\]

如果词表大小为 \(V\)，则需要对大量候选词计算：

\[
DV
\]

Hierarchical Softmax 将整个词表组织成一棵 Huffman Binary Tree。

预测一个目标词时，只需要沿：

```text
root
 ↓
node
 ↓
node
 ↓
target word
```

这条路径进行若干次二分类。

路径长度约为：

\[
\log_2V
\]

因此复杂度降低为：

\[
D\log_2V
\]

每个内部节点都会计算：

\[
h^Tu_j
\]

并通过 sigmoid 得到分支概率。

一个词的概率可写成：

\[
P(w|h)
=
\prod_{j\in path(w)}
P(branch_j|h)
\]

因此，训练时不需要对整个词表的所有词逐个计算分数。

---

# 7. Evaluation

论文不只通过“最相似词”来评价 embedding，而是设计 **word analogy task**。

例如：

\[
big:bigger=small:?
\]

计算：

\[
v(bigger)-v(big)+v(small)
\]

然后在向量空间中寻找最接近的词。

理想结果：

\[
smaller
\]

类似的语义关系还有：

\[
France:Paris=Germany:Berlin
\]

论文构造了 **Semantic-Syntactic Word Relationship Test Set**，包括：

- 5 类 semantic relationships；
- 9 类 syntactic relationships；
- 8869 个 semantic questions；
- 10675 个 syntactic questions。

评价采用 exact match：

> 只有最近的词与标准答案完全一致才算正确。

---

# 8. Experimental Results

训练语料主要使用 Google News Corpus：

\[
6B\ tokens
\]

词表最大约：

\[
1M\ words
\]

实验主要研究：

- 训练数据量；
- embedding 维度；
- 不同模型架构；
- epoch 数；
- 大规模并行训练。

---

## 8.1 More Data + Larger Dimension

实验表明：

- 增加训练数据通常可以提高词向量质量；
- 增加 embedding dimension 通常也可以提高效果；
- 单独持续增加某一个因素会出现 diminishing returns。

因此较合理的做法是：

\[
\boxed{Training\ Data\uparrow + Embedding\ Dimension\uparrow}
\]

而不是只增大其中一个。

---

## 8.2 Model Comparison

在相同数据和相同 640 维 embedding 下：

| Model | Semantic Accuracy | Syntactic Accuracy |
|---|---:|---:|
| RNNLM | 9% | 36% |
| NNLM | 23% | 53% |
| CBOW | 24% | **64%** |
| Skip-gram | **55%** | 59% |

可以看出：

- CBOW 在 syntactic tasks 上更强；
- Skip-gram 在 semantic tasks 上明显更强。

---

## 8.3 More Data vs More Epochs

论文发现：

> 使用更多不同的数据训练一次，往往可以达到甚至超过在较小数据集上重复训练多次的效果。

这体现了整篇论文的核心思路：

\[
\boxed{Simple\ Model + More\ Data}
\]

---

## 8.4 Large-scale Training

作者使用 DistBelief 在 Google News 6B tokens 上进行大规模并行训练。

其中 CBOW 和 Skip-gram 可以训练到：

\[
1000\text{-dimensional embeddings}
\]

并保持较高准确率。

这验证了论文的核心观点：

```text
模型更简单
   ↓
单个样本计算成本更低
   ↓
可以使用更大数据、更高维 embedding
   ↓
最终词向量质量更高
```

---

# 9. Learned Vector Relationships

训练后的 embedding 能表现出一定的线性语义和句法规律。

例如：

\[
Paris-France+Italy\approx Rome
\]

以及：

```text
Miami : Florida
Dallas : Texas
```

```text
big : bigger
small : smaller
```

说明词向量不仅能够表达：

\[
word\ similarity
\]

还能够在一定程度上表达：

\[
semantic/syntactic\ relationships
\]

论文还发现，使用多个关系样例的平均向量，可以进一步提高 analogy accuracy。

---

# 10. Core Contributions

## 10.1 提出 CBOW 和 Skip-gram

两个结构简单、高效的词向量学习模型：

\[
CBOW
\]

\[
Skip\text{-}gram
\]

---

## 10.2 大幅降低训练成本

通过去除传统神经语言模型中的复杂 nonlinear hidden layer，使模型能够使用：

- 更大的训练语料；
- 更大的词表；
- 更高维的 embedding。

---

## 10.3 展示词向量中的线性规律

词向量空间中能够出现类似：

\[
King-Man+Woman\approx Queen
\]

这样的关系。

这说明 embedding space 可以编码一定的语义和句法结构。

---

# 11. Relation to Modern NLP

Word2Vec 学习的是：

\[
\boxed{Static\ Word\ Embedding}
\]

同一个词无论出现在哪个上下文中，都对应同一个基础 embedding。

例如：

```text
I went to the bank to withdraw money.

I sat on the bank of the river.
```

Word2Vec 中两个 `bank` 对应同一个词向量。

而后来的 Transformer / BERT 更强调：

\[
\boxed{Contextual\ Representation}
\]

同一个 token 在不同上下文中，经过上下文建模后会得到不同表示。

可以将表示学习的发展粗略理解为：

```text
Discrete Symbol / One-hot
        ↓
Word2Vec
Static Word Embedding
        ↓
RNN / LSTM
Sequence Representation
        ↓
Attention
        ↓
Transformer / BERT
Contextual Representation
```

需要注意：

> Word2Vec 并不是 Transformer 的直接架构前身，但它推动了“用稠密向量学习语言表示”这一范式的发展。

---

# 12. Takeaway

整篇论文可以压缩成一句话：

> **通过去掉传统神经语言模型中昂贵的 nonlinear hidden layer，CBOW 和 Skip-gram 可以在超大规模文本上高效学习高质量词向量，并使语义和句法关系在向量空间中呈现一定的线性规律。**

核心主线：

```text
Discrete Word Representation
        ↓
Continuous Embedding
        ↓
NNLM / RNNLM 计算成本高
        ↓
Remove Complex Hidden Layer
        ↓
CBOW / Skip-gram
        ↓
Large-scale Training
        ↓
Semantic & Syntactic Regularities
```
