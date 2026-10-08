# Efficient Estimation of Word Representations in Vector Space

> Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean, 2013  
> Keywords: Word2Vec, CBOW, Skip-gram, Word Embedding

## 先回答五个问题

### 这篇论文解决了什么问题？

这篇论文关注的是：**如何在超大规模语料上，以较低计算成本学习高质量的词向量（word representations / word embeddings）**。

作者希望学到的词向量不仅能表示“词和词是否相似”，还能够体现一定的语义和句法关系，例如：

```text
big : bigger = small : smaller
Paris : France = Rome : Italy
```

### 以前的方法有什么问题？为什么要解决这个问题？

当时很多 NLP 系统把词当成词表中的独立符号或 index，本身并不包含词之间的相似关系。

神经网络语言模型虽然能够学习连续词向量，但计算成本较高：

- Feedforward NNLM 中，Projection → Hidden 的计算很大；
- RNNLM 中，Hidden → Hidden 的循环计算很大；
- 当词表和训练语料变得很大时，训练成本会迅速上升。

因此，作者认为与其使用复杂模型处理较少数据，不如设计**更简单、更高效的模型**，从而利用更大规模的数据训练词向量。

### 使用了什么方法解决？

作者提出两个简单的 log-linear 模型：

- **CBOW**：根据上下文词预测中心词；
- **Skip-gram**：根据中心词预测周围的上下文词。

核心思路是：

```text
去掉传统神经语言模型中昂贵的 nonlinear hidden layer
                ↓
降低单个训练样本的计算成本
                ↓
能够使用更大的语料和更高维的词向量
```

论文还使用 **Hierarchical Softmax** 降低大词表下的输出计算成本。

### 结果是什么？

实验表明：

- CBOW 和 Skip-gram 能够以较低计算成本训练高质量词向量；
- CBOW 在论文的 syntactic analogy 任务上表现更好；
- Skip-gram 在 semantic analogy 任务上表现尤其突出；
- 增加训练数据量和词向量维度通常能够提高效果；
- 相比在同一批数据上重复训练，使用更多不同的数据往往更有效；
- 学到的词向量能够表现出一定的线性语义和句法规律。

例如：

```text
Paris - France + Italy ≈ Rome
```

### 未来展望是什么？

作者认为高质量词向量会成为未来 NLP 系统的重要基础组件，并提到其潜在应用包括：

- sentiment analysis；
- paraphrase detection；
- machine translation；
- knowledge base extension；
- knowledge base fact verification。

论文还认为，更大的训练语料、更高维的词向量以及更好的关系建模方法仍有进一步提升空间。

---

# 1. Introduction

论文首先指出，当时很多 NLP 方法将词视为彼此独立的离散符号。

例如：

```text
cat -> 127
dog -> 45
```

这些 index 只表示“这是哪个词”，并不能直接表示：

```text
cat 与 dog 比 cat 与 table 更相似
```

连续词表示的目标，是将词映射为低维稠密向量：

```text
word -> embedding
```

词之间的语义和句法关系，可以通过这些向量在空间中的位置和方向体现出来。

## 1.1 Goals of the Paper

作者的主要目标是：

> 在包含数十亿词的大规模数据集上学习高质量词向量，并支持百万级词表。

作者不仅关心相似词是否靠近，还关心词向量是否能够表现更复杂的关系，例如：

```text
King - Man + Woman ≈ Queen
```

论文希望通过新的模型结构，在**准确率和计算效率之间取得更好的平衡**。

## 1.2 Previous Work

连续词向量并不是本文首次提出。

此前的 NNLM、RNNLM 等方法已经可以在训练语言模型的同时学习词向量。

本文与这些工作的区别在于：

> **重点不再是构建更复杂的语言模型，而是尽可能高效地学习高质量词向量。**

---

# 2. Model Architectures

作者先分析已有神经网络语言模型的计算成本。

训练复杂度可以粗略写成：

```text
总训练复杂度 = epoch 数 × 训练词数 × 单个样本计算成本
```

因为论文希望使用非常大的训练语料，所以关键是降低**单个训练样本的计算成本**。

## 2.1 Feedforward Neural Net Language Model (NNLM)

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

主要符号：

- `N`：上下文词数量；
- `D`：词向量维度；
- `H`：hidden layer 大小；
- `V`：词表大小。

计算成本主要来自：

```text
Projection -> Hidden
```

以及：

```text
Hidden -> Vocabulary
```

其中输出层可以通过 Hierarchical Softmax 降低成本，但复杂 hidden layer 仍然很昂贵。

## 2.2 Recurrent Neural Net Language Model (RNNLM)

RNN 通过上一时刻的 hidden state 保存历史信息，因此不需要固定上下文长度。

但它需要反复进行：

```text
hidden(t-1) -> hidden(t)
```

的矩阵计算。

因此，RNNLM 的主要计算瓶颈同样来自 hidden layer。

## 2.3 Parallel Training of Neural Networks

为了处理大规模数据，作者使用 DistBelief 进行并行训练。

多个模型副本并行处理数据，并通过中心参数服务器同步更新。

这一部分主要说明：

> 大规模并行能够加速训练，但如果模型本身过于复杂，计算成本仍然很高。

---

# 3. New Log-linear Models

这一部分是论文的核心。

作者根据 Section 2 的分析发现：

> 大量计算成本来自 nonlinear hidden layer。

因此提出一个直接的思路：

```text
不追求复杂模型
      ↓
去掉 nonlinear hidden layer
      ↓
让模型更简单、更快
      ↓
用更多数据训练
```

## 3.1 Continuous Bag-of-Words Model (CBOW)

CBOW 的任务是：

```text
Context -> Center Word
```

例如：

```text
The cat [ ? ] sitting on the mat
```

模型利用：

```text
The, cat, sitting, on
```

预测：

```text
is
```

多个上下文词的 embedding 会被组合后用于预测中心词。

CBOW 不考虑上下文中词的顺序，因此称为 Bag-of-Words。

它最大的特点是：

> **结构简单，计算速度快。**

## 3.2 Continuous Skip-gram Model

Skip-gram 与 CBOW 相反：

```text
Center Word -> Context
```

例如：

```text
I like drinking hot coffee every morning
```

以：

```text
coffee
```

作为输入，预测它附近的：

```text
hot
drinking
every
morning
```

窗口越大，可以利用更多上下文，但计算成本也会增加。

论文还采用随机上下文窗口，使距离中心词更近的词更频繁地参与训练。

## CBOW 与 Skip-gram

| 项目 | CBOW | Skip-gram |
|---|---|---|
| 输入 | 多个上下文词 | 一个中心词 |
| 输出 | 中心词 | 多个上下文词 |
| 方向 | Context → Word | Word → Context |
| 特点 | 更高效 | 需要更多预测 |
| 论文实验表现 | syntactic 较强 | semantic 较强 |

---

# 4. Results

作者没有单独设置一个完整的 “Experimental Setup” 章节，而是把实验设置、评测任务和结果都放在 Results 中。

## 4.1 Task Description

作者构建了一个 **Semantic-Syntactic Word Relationship Test Set**。

包括：

- 5 类 semantic relationships；
- 9 类 syntactic relationships；
- 8869 个 semantic questions；
- 10675 个 syntactic questions。

评价方式是 word analogy。

例如：

```text
big : bigger = small : ?
```

模型利用词向量关系寻找答案：

```text
small -> smaller
```

语义关系的例子包括：

```text
France : Paris = Germany : Berlin
```

评价采用 exact match：只有最近的词与标准答案完全一致才算正确。

## 4.2 Maximization of Accuracy

作者使用 Google News Corpus：

```text
约 6B tokens
```

词表规模最高达到：

```text
约 1M words
```

实验主要研究两个因素：

- 训练数据量；
- embedding dimension。

结果表明：

> 增加数据量和向量维度通常都能提高准确率，但单独持续增加其中一个会出现收益递减。

因此作者建议二者一起增加。

## 4.3 Comparison of Model Architectures

作者比较了：

- RNNLM；
- NNLM；
- CBOW；
- Skip-gram。

在相同训练数据和 640 维词向量下：

| Model | Semantic Accuracy | Syntactic Accuracy |
|---|---:|---:|
| RNNLM | 9% | 36% |
| NNLM | 23% | 53% |
| CBOW | 24% | **64%** |
| Skip-gram | **55%** | 59% |

可以看出：

- CBOW 在 syntactic task 上表现更强；
- Skip-gram 在 semantic task 上明显更好。

论文还发现：

> 使用更多不同的数据训练一次，往往能达到或超过在较少数据上重复训练多个 epoch 的效果。

这与论文的核心思想一致：

```text
Simple Model + More Data
```

## 4.4 Large Scale Parallel Training of Models

作者进一步使用 DistBelief 在约 6B tokens 上训练模型。

CBOW 和 Skip-gram 可以扩展到：

```text
1000-dimensional word vectors
```

并取得比传统 NNLM 更好的结果。

这证明：

```text
简单模型
  ↓
计算成本降低
  ↓
可以使用更大数据和更高维表示
  ↓
最终得到更好的词向量
```

## 4.5 Microsoft Research Sentence Completion Challenge

作者还在句子补全任务上测试 Skip-gram。

任务形式是：

```text
一个句子缺少一个词
+
5 个候选词
```

Skip-gram 单独并没有超过最佳 RNNLM，但：

```text
Skip-gram + RNNLM
```

取得了更好的结果。

说明 Skip-gram 学到的信息与 RNNLM 具有一定互补性。

---

# 5. Examples of the Learned Relationships

这一节主要展示训练后词向量中的实际关系。

例如：

```text
Paris - France + Italy ≈ Rome
```

还有：

```text
Miami : Florida
Dallas : Texas
```

以及：

```text
big : bigger
small : smaller
```

这些结果说明：

> 词向量不仅能够编码“词是否相似”，还可以在一定程度上编码语义和句法关系。

作者还发现：

> 使用多个关系样例的平均向量，比只使用一个关系样例更稳定，可以进一步提高 analogy accuracy。

论文还提到，词向量可以用于：

```text
从一组词中找出不属于同一类别的词
```

说明这种向量空间结构还可以支持其他任务。

---

# 6. Conclusion

论文最后总结：

> **非常简单的模型也可以训练出高质量词向量。**

与复杂的 feedforward / recurrent neural network 相比，CBOW 和 Skip-gram 的计算复杂度更低，因此能够：

- 使用更大的训练数据；
- 使用更高维的词向量；
- 支持更大的词表。

作者认为高质量词向量未来会成为 NLP 系统中的重要基础组件。

论文提到的潜在应用包括：

- sentiment analysis；
- paraphrase detection；
- machine translation；
- knowledge base extension；
- knowledge base fact verification。

整篇论文最核心的思想可以总结为：

```text
传统神经语言模型能学词向量
        ↓
但是计算成本高
        ↓
去掉昂贵的 nonlinear hidden layer
        ↓
CBOW / Skip-gram
        ↓
用更大的语料训练
        ↓
得到高质量的 word embeddings
```

---

# 7. Follow-Up Work

论文最后补充了后续工作。

作者发布了单机多线程 C++ 实现，支持：

- CBOW；
- Skip-gram。

新的实现速度比论文早期实验中报告的速度更快。

作者还发布了大规模预训练词向量，并表示后续工作会进一步研究：

- 更大规模语料；
- 更高效的词向量训练；
- word / phrase representation 的进一步改进。
