# Efficient Estimation of Word Representations in Vector Space：论文复习笔记

> **Authors:** Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean  
> **Year:** 2013  
> **arXiv:** [1301.3781](https://arxiv.org/abs/1301.3781)  
> **Keywords:** Word2Vec, CBOW, Skip-gram, Word Embedding

> 注意：这篇论文提出 CBOW 和 Skip-gram，并主要使用 hierarchical softmax。常与 Word2Vec 一起讨论的 negative sampling 和高频词 subsampling 来自后续论文，不应算作本文贡献。

## 论文概览

| 问题 | 结论 |
|---|---|
| 解决什么问题？ | 在十亿词级语料和大词表上，以较低成本学习高质量的稠密词向量。 |
| 为什么要解决？ | 传统 one-hot/index 不表达词间关系；NNLM、RNNLM 虽能学习词向量，但非线性隐藏层和大词表输出层计算昂贵。 |
| 使用什么方法？ | 去掉昂贵的非线性隐藏层，提出 CBOW（上下文预测中心词）和 Skip-gram（中心词预测上下文），并用 hierarchical softmax 降低输出成本。 |
| 结果是什么？ | 简单模型能利用更多数据和更高维向量，在语义/句法类比测试上明显优于多种已有词向量；Skip-gram 尤其擅长语义关系，CBOW 训练更快。 |
| 未来展望是什么？ | 扩大训练语料和词表，将词向量用于更多 NLP 与知识库任务，并继续改进训练效率及 word/phrase representation。 |

论文的核心取向不是“模型越复杂越好”，而是：

```text
简化模型结构
→ 降低每个训练样本的成本
→ 使用更多数据和更高维向量
→ 学到更好的词表示
```

## 1. Introduction

很多传统 NLP 系统把词视为彼此独立的离散符号：

```text
cat → 127
dog → 45
```

这些编号只能说明“这是哪个词”，不能直接表达 **cat 与 dog 比 cat 与 table 更相似**。

连续词表示则把词映射为低维稠密向量。训练良好时，相似词在向量空间中更接近，某些语义和句法关系还可能表现为近似一致的方向。

### 1.1 Goals of the Paper

作者的目标是：

- 在包含数十亿词的语料上训练词向量；
- 支持百万级词表；
- 在计算成本可控的情况下提高向量维度；
- 不只评估“相似词是否接近”，还评估关系能否通过向量运算保持。

典型例子是：

```math
v(\mathrm{King})-v(\mathrm{Man})+v(\mathrm{Woman})
\approx v(\mathrm{Queen})
```

论文更系统地测试两类关系：

- **语义关系**：国家—首都、货币、城市—州、男性—女性等；
- **句法关系**：比较级、最高级、过去式、复数、形容词—副词等。

### 1.2 Previous Work

连续词向量并非本文首次提出。此前的 feedforward NNLM、RNNLM 和其他表示学习方法已经能训练词向量。

本文的区别是：

> 不把“构建更强的完整语言模型”作为主要目标，而是直接优化“怎样更高效地学出高质量词向量”。

作者还强调，一个词可能具有多种相似性。本文仍为每个词学习单一向量，并未真正解决一词多义；这属于阅读时需要保留的限制。

## 2. Model Architectures

论文用下面的统一形式衡量训练复杂度：

```math
O=E\times T\times Q
```

其中：

- $E$：训练 epoch 数；
- $T$：训练语料中的词数；
- $Q$：每个训练样本的计算成本。

由于 $T$ 可能达到十亿级，降低 $Q$ 是扩展到大语料的关键。

### 2.1 Feedforward Neural Net Language Model (NNLM)

典型结构是：

```text
前 N 个词
→ Projection
→ Non-linear Hidden Layer
→ Output Vocabulary
```

符号：

- $N$：上下文词数；
- $D$：词向量维度；
- $H$：隐藏层大小；
- $V$：词表大小。

每个训练样本的复杂度为：

```math
Q=N\times D+N\times D\times H+H\times V
```

普通 softmax 的 $H\times V$ 在大词表下非常昂贵。论文使用基于 Huffman 树的 hierarchical softmax，将输出层需要评估的节点数从 $V$ 降到约 $\log_2(V)$。

使用 hierarchical softmax 后，主要瓶颈通常变成 Projection 到 Hidden 的 $N\times D\times H$。

### 2.2 Recurrent Neural Net Language Model (RNNLM)

RNNLM 使用循环隐藏状态保存历史信息，因此不需要预先固定上下文长度：

```math
h_t=f(h_{t-1},x_t)
```

每个训练样本的复杂度近似为：

```math
Q=H\times H+H\times V
```

hierarchical softmax 可把 $H\times V$ 降为约 $H\times\log_2(V)$，但循环隐藏层仍需进行 $H\times H$ 的矩阵计算。

### 2.3 Parallel Training of Neural Networks

作者使用 DistBelief 进行大规模分布式训练：

- 同一模型的多个副本并行处理数据；
- 每个副本使用 mini-batch 异步梯度下降；
- 中心参数服务器保存并更新参数；
- 实验可使用 100 个以上模型副本和大量 CPU 核心。

这一节说明分布式系统可以扩大训练规模，但模型本身的单样本复杂度仍然决定整体效率。

## 3. New Log-linear Models

Section 2 表明，NNLM 和 RNNLM 的大部分成本来自非线性隐藏层。作者因此提出两个没有非线性隐藏层的 log-linear 模型。

### 3.1 Continuous Bag-of-Words Model (CBOW)

CBOW 根据上下文预测中心词：

```text
w(t-2), w(t-1), w(t+1), w(t+2)
                  ↓
                w(t)
```

上下文词的向量在 projection 层求和或平均，再用于预测中心词。因为不考虑这些上下文词的内部顺序，所以称为 Bag-of-Words。

在论文设置中，CBOW 使用前后各 4 个词预测中间词。每个样本的复杂度为：

```math
Q=N\times D+D\times\log_2(V)
```

其中：

- $N\times D$：组合 $N$ 个上下文词向量；
- $D\times\log_2(V)$：通过 hierarchical softmax 预测中心词。

CBOW 的优点是一次组合多个上下文来完成一次预测，训练速度快；代价是求和/平均会忽略上下文顺序并压缩细节。

### 3.2 Continuous Skip-gram Model

Skip-gram 的方向与 CBOW 相反：使用中心词预测周围词。

```text
             w(t-2)
             w(t-1)
w(t)   →    w(t+1)
             w(t+2)
```

其复杂度为：

```math
Q=C\times\left(D+D\times\log_2(V)\right)
```

$C$ 是最大上下文距离。论文实验中使用 $C=10$。

实际训练时，从 $1$ 到 $C$ 随机选择窗口半径 $R$，再预测中心词左右各 $R$ 个词。由于近处词在更多可能窗口中出现，它们被采样得更频繁；远处词通常与中心词关系更弱，也会获得较低的训练频率。

### CBOW 与 Skip-gram

| | CBOW | Skip-gram |
|---|---|---|
| 输入 | 多个上下文词 | 一个中心词 |
| 预测目标 | 中心词 | 多个上下文词 |
| 单个中心位置的预测次数 | 1 | 多次 |
| 训练速度 | 通常更快 | 通常更慢 |
| 本文实验特点 | 句法类比表现较强 | 语义类比表现突出 |

不要把这张表理解成普遍定律。结果会受到语料、词频、窗口、维度和训练目标等因素影响。

## 4. Results

### 4.1 Task Description

作者构建了 **Semantic-Syntactic Word Relationship Test Set**：

- 5 类语义关系；
- 9 类句法关系；
- 8,869 个语义问题；
- 10,675 个句法问题。

类比问题形如：

```text
big : biggest = small : ?
```

计算方式是：

```math
x=v(\mathrm{biggest})-v(\mathrm{big})+v(\mathrm{small})
```

然后使用 cosine distance 在词表中寻找最接近 $x$ 的词，期望答案为 **smallest**。搜索时会排除问题中已经出现的词。

评价采用 exact match：

- 最近邻必须与标准答案完全相同才算正确；
- 同义词也会被判错；
- 测试集只包含单 token 词，不含 New York 之类的多词实体；
- 模型没有显式词形信息，因此 100% 准确率并不现实。

因此，论文所谓 “word similarity” 更准确地说是 **语义/句法关系类比测试**，不是普通的词对相似度评分。

### 4.2 Maximization of Accuracy

作者使用约 60 亿 token 的 Google News 语料，并把词表限制为最常见的 100 万词。

Table 2 使用 CBOW 研究训练数据量和向量维度。主要趋势是：

- 数据更多通常提高准确率；
- 向量维度更高通常提高准确率；
- 单独持续增加其中一项会出现收益递减；
- 数据规模和向量维度最好一起提高。

例如在该子集实验中，600 维 CBOW 从 24M 训练词的 24.0% 提升到 783M 训练词的 50.4%。

Table 2 与 Table 4 的实验训练 3 个 epoch，使用 SGD 和反向传播；初始学习率为 0.025，并在训练过程中线性衰减到接近 0。

### 4.3 Comparison of Model Architectures

在相同的 320M 训练词和 640 维向量下，Table 3 给出：

| Model | Semantic | Syntactic | MSR Relatedness |
|---|---:|---:|---:|
| RNNLM | 9% | 36% | 35% |
| NNLM | 23% | 53% | 47% |
| CBOW | 24% | **64%** | **61%** |
| Skip-gram | **55%** | 59% | 56% |

结论：

- CBOW 在这组实验的句法测试上最好；
- Skip-gram 的语义类比明显更强；
- 简化模型没有导致词向量质量下降，反而允许使用更多数据。

Table 4 在完整词表上比较公开词向量。300 维 Skip-gram 使用 783M 训练词时达到：

- 语义：50.0%；
- 句法：55.9%；
- 总准确率：53.3%。

Table 5 进一步表明，在相同计算预算附近，使用更多不同数据训练一个 epoch，往往优于在较少数据上重复多个 epoch。例如：

- 3 epoch Skip-gram，783M 词：53.3%，约 3 天；
- 1 epoch Skip-gram，1.6B 词：53.8%，约 2 天。

这支持论文的核心取向：**Simple Model + More Data**。

### 4.4 Large Scale Parallel Training of Models

作者使用 DistBelief 在约 6B token 上训练 1000 维向量。Table 6 的结果为：

| Model | Dimensions | Semantic | Syntactic | Total | Training |
|---|---:|---:|---:|---:|---:|
| NNLM | 100 | 34.2% | 64.5% | 50.8% | 14 days × 180 cores |
| CBOW | 1000 | 57.3% | 68.9% | 63.7% | 2 days × 140 cores |
| Skip-gram | 1000 | **66.1%** | 65.1% | **65.6%** | 2.5 days × 125 cores |

这支持 CBOW/Skip-gram 在大规模语料上的效率优势，但比较并不完全等价：

- 模型维度不同；
- CPU 使用量是估计值；
- 数据中心机器同时承担其他任务；
- 分布式框架存在额外通信开销。

### 4.5 Microsoft Research Sentence Completion Challenge

该任务包含 1,040 个句子，每个句子缺少一个词，并提供 5 个候选答案。

Table 7：

| Model | Accuracy |
|---|---:|
| 4-gram | 39.0% |
| Average LSA similarity | 49.0% |
| Log-bilinear model | 54.8% |
| RNNLMs | 55.4% |
| Skip-gram | 48.0% |
| Skip-gram + RNNLMs | **58.9%** |

Skip-gram 单独不如 RNNLM，因为它主要学习局部词关系，不是完整句子概率模型。但两者结合后超过 RNNLM，说明它们提供的信息具有互补性。

## 5. Examples of the Learned Relationships

论文展示了多种向量关系：

- France : Paris → Italy : Rome；
- big : bigger → small : larger；
- Miami : Florida → Dallas : Texas；
- Einstein : scientist → Mozart : violinist；
- Microsoft : Windows → Google : Android。

关系向量的一般形式为：

```math
r=v(b)-v(a)
```

再把它加到另一个词向量上：

```math
x=v(c)+r=v(c)+v(b)-v(a)
```

最后寻找与 $x$ 最接近的词。

这些例子说明词向量中存在一定的线性规律，但不应过度解读：

- Table 8 的人工示例并非全部能通过严格 exact-match 指标答对；
- 论文指出这些示例若按其准确率指标计算，得分约为 60%；
- 使用 10 个关系样例的平均向量代替单一关系向量，可使最佳模型在测试集上提高约 10 个百分点；
- 向量规律是统计性的，不是严格代数等式。

论文还提到可以用平均向量和最远距离寻找一组词中的离群词，说明词向量可支持类比以外的任务。

## 6. Conclusion

论文的核心结论是：

> 与复杂 NNLM/RNNLM 相比，结构非常简单的 CBOW 和 Skip-gram 也能学习高质量词向量；更低的计算成本使其能够利用更大的数据、更高的维度和更大的词表。

主要贡献包括：

1. 提出 CBOW 和 Skip-gram 两种高效 log-linear 架构；
2. 系统比较模型复杂度、数据规模和向量维度；
3. 构建包含语义与句法关系的类比测试集；
4. 展示词向量中的线性关系及其潜在应用；
5. 证明“简单模型 + 大规模数据”可以优于更复杂但训练规模较小的方法。

论文提到的应用包括情感分析、复述检测、机器翻译、信息检索、问答、知识库扩展和事实验证。

### 阅读时需要保留的边界

- 每个词只有一个静态向量，不能根据上下文改变含义；
- 没有 subword 机制，罕见词和词表外词处理能力有限；
- 类比测试只覆盖预设关系，且 exact match 会忽略合理同义答案；
- 向量算术展示的是统计规律，不表示模型真正掌握了形式逻辑；
- 训练语料中的偏差也可能进入词向量。

## 7. Follow-Up Work

论文补充的后续工作包括：

- 发布单机多线程 C++ 版本的 CBOW 和 Skip-gram；
- 典型配置下训练速度达到每小时数十亿词，比论文早期分布式实验更快；
- 发布超过 140 万个词和命名实体的向量；
- 这些向量使用超过 1,000 亿词训练；
- 后续工作将进一步研究 word/phrase representation 及其组合性。

Negative sampling、常见词 subsampling 和短语学习通常与 Word2Vec 一起讨论，但主要在后续论文中系统提出，不应回写成本文已经完成的贡献。

## Reference

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). **Efficient Estimation of Word Representations in Vector Space.** arXiv:1301.3781.
