# Attention Is All You Need — 复习笔记

> Vaswani et al., NIPS 2017  
> 核心目标：用 **Self-Attention** 替代 RNN/CNN，提升并行能力并缩短长距离依赖路径。

---

## 1. 论文核心

传统序列模型主要依赖 RNN/LSTM/GRU 或 CNN。

### RNN 的问题
RNN 中：

$$
h_t = f(h_{t-1}, x_t)
$$

当前位置依赖前一位置，因此难以并行。

### CNN 的问题
CNN 可并行，但远距离 token 需要经过多层才能交互。

### Transformer 的思路
直接使用 Self-Attention，让任意两个位置直接建立联系。

---

## 2. Transformer 结构

整体仍是：

```text
Encoder → Decoder
```

原论文中：

$$
N=6
$$

### Encoder Layer

```text
Multi-Head Self-Attention
        ↓
    Add & Norm
        ↓
       FFN
        ↓
    Add & Norm
```

### Decoder Layer

```text
Masked Self-Attention
        ↓
    Add & Norm
        ↓
Encoder-Decoder Attention
        ↓
    Add & Norm
        ↓
       FFN
        ↓
    Add & Norm
```

Decoder 要加 Mask，因为生成第 $i$ 个 token 时不能看到未来 token；Encoder 输入本身完整已知，所以不需要 causal mask。

---

## 3. Attention：Q、K、V

- **Query**：当前想找什么信息
- **Key**：用什么判断是否相关
- **Value**：真正被提取的信息

核心公式：

$$
Attention(Q,K,V)
=
softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

步骤：

1. $QK^T$：计算匹配分数
2. 除以 $\sqrt{d_k}$：控制点积尺度，避免 softmax 过度饱和
3. softmax：转成注意力权重
4. 乘 $V$：得到加权信息

---

## 4. Multi-Head Attention

$$
MultiHead(Q,K,V)
=
Concat(head_1,\ldots,head_h)W^O
$$

其中：

$$
head_i
=
Attention(QW_i^Q,KW_i^K,VW_i^V)
$$

不同 Head 使用不同投影矩阵，因此可以学习不同关系。

原论文 Base：

- $d_{model}=512$
- $h=8$
- $d_k=d_v=64$

可记为：

$$
512 \rightarrow 8\times64 \rightarrow 512
$$

注意：不是简单把 512 维切成 8 段，而是每个 Head 通过独立线性投影得到 64 维表示。

---

## 5. 三种 Attention

| 类型 | Q 来源 | K/V 来源 |
|---|---|---|
| Encoder Self-Attention | Encoder | Encoder |
| Decoder Masked Self-Attention | Decoder | Decoder |
| Encoder-Decoder Attention | Decoder | Encoder |

---

## 6. FFN

每个位置独立经过同一个前馈网络：

$$
FFN(x)
=
max(0,xW_1+b_1)W_2+b_2
$$

原论文：

- $d_{model}=512$
- $d_{ff}=2048$

即：

```text
512 → 2048 → ReLU → 512
```

可以简单理解：

- Attention：不同 token 交换信息
- FFN：每个 token 自己进一步加工

---

## 7. Positional Encoding

Transformer 没有 RNN/CNN，因此需要额外加入位置信息：

$$
Input = Embedding + PE
$$

PE 与 Embedding 维度相同：

$$
d_{model}=512
$$

论文使用：

$$
PE(pos,2i)
=
\sin\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

$$
PE(pos,2i+1)
=
\cos\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

- $pos$：token 位置
- 偶数维：sin
- 奇数维：cos

论文也测试了 learned positional embedding，效果与 sinusoidal PE 接近。

---

## 8. 关键维度

| 符号 | Base | 含义 |
|---|---:|---|
| $d_{model}$ | 512 | token 主干表示维度 |
| $h$ | 8 | Head 数 |
| $d_k$ | 64 | 单个 Head 的 Q/K 维度 |
| $d_v$ | 64 | 单个 Head 的 V 维度 |
| $d_{ff}$ | 2048 | FFN 中间层维度 |

---

## 9. 为什么 Self-Attention？

| Layer | Complexity | Sequential Ops | Max Path Length |
|---|---:|---:|---:|
| Self-Attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| RNN | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| CNN | $O(knd^2)$ | $O(1)$ | $O(\log_k n)$ |

### Self-Attention
每个 token 和所有 token 计算关系：

$$
O(n^2d)
$$

### RNN
每个时间步有一个大约 $d\times d$ 的隐藏状态变换：

$$
O(nd^2)
$$

### CNN
每个位置看 $k$ 个邻域位置，并进行 $d\rightarrow d$ 的通道映射：

$$
O(knd^2)
$$

当典型机器翻译中 $n<d$ 时，Self-Attention 的计算成本有竞争力；更重要的是它可高度并行，且任意两个 token 的路径长度为 $O(1)$。

---

## 10. Training

### 数据
- WMT 2014 EN-DE：约 4.5M sentence pairs
- WMT 2014 EN-FR：约 36M sentence pairs

### 优化
- Adam
- warmup steps = 4000
- 学习率：先升高，再按步数平方根倒数衰减

$$
lrate
=
d_{model}^{-0.5}
\cdot
\min(
step^{-0.5},
step\cdot warmup^{-1.5}
)
$$

### Regularization
- Dropout
- Label Smoothing

---

## 11. Base vs Big

| 参数 | Base | Big |
|---|---:|---:|
| $N$ | 6 | 6 |
| $d_{model}$ | 512 | 1024 |
| $d_{ff}$ | 2048 | 4096 |
| Heads | 8 | 16 |
| Parameters | ~65M | ~213M |
| Steps | 100K | 300K |

Big 不是另一种 Transformer，而是更大的同一架构。

---

## 12. 实验结论

### Machine Translation
- Transformer Base：27.3 BLEU
- Transformer Big：28.4 BLEU（EN-DE）

### Model Variations
论文发现：

- 单头 Attention 更差
- Head 太多也不一定更好
- $d_k$ 太小会降低性能
- 更大的模型通常更强
- Dropout 有助于防止过拟合
- Sinusoidal PE 与 learned PE 效果接近

### Generalization
Transformer 在 English Constituency Parsing 上同样表现很好，说明其不只适用于机器翻译。

---

## 13. 核心创新点

1. 用 Self-Attention 替代传统 recurrence / convolution
2. 提出 Multi-Head Attention
3. 使用 Positional Encoding 补充顺序信息
4. 提升并行能力
5. 缩短长距离依赖路径

---

## 14. 一分钟复习

```text
Transformer = Attention-based Encoder-Decoder

Encoder:
Self-Attention + FFN

Decoder:
Masked Self-Attention
+ Cross-Attention
+ FFN

核心公式：
Attention(Q,K,V)
= softmax(QKᵀ / √dk)V

Base:
dmodel = 512
heads = 8
dff = 2048

PE:
Embedding + Positional Encoding

Self-Attention:
O(n²d), Sequential O(1), Path O(1)

RNN:
O(nd²), Sequential O(n), Path O(n)

核心思想：
去掉 RNN/CNN，
用 Self-Attention 建模序列。
```

---

## Reference

Vaswani, A. et al. **Attention Is All You Need.** NIPS 2017.
