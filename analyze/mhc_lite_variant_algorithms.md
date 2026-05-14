



# mHC-lite 三种改进方法的流程图与数学公式

这是一份 PDF 友好的 Markdown 版本。

- 不使用 `$$ ... $$` 数学环境
- 不使用 Mermaid
- 公式全部改成普通代码块中的 ASCII 写法
- 流程图全部改成纯文本流程图

这样可以避免 Markdown 转 PDF 时因为缺少 MathJax、KaTeX、Mermaid 插件而导致渲染失败。

本文档对应当前仓库中新增的三个 `mhc_lite_method`:

- `selective`
- `depth_attn`
- `block_attn`

对应代码:

- `selective`: `hyper_conn/mhc_lite.py`
- `depth_attn`: `hyper_conn/attention_residuals.py` + `model.py`
- `block_attn`: `hyper_conn/attention_residuals.py` + `model.py`

参考论文:

- [mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations](https://arxiv.org/abs/2601.05732)
- [Attention Residuals](https://arxiv.org/abs/2603.15031)

## 统一记号

设:

- `S`: 残差流数量
- `d`: 单个流的特征维度
- `X_t`: 第 `t` 个子层输入的多流表示
- `x_t^(i)`: 第 `t` 个子层第 `i` 条流的表示
- `xhat_t = vec(X_t)`: 展平后的向量
- `xbar_t = RMSNorm(xhat_t)`: 归一化后的向量
- `{P_r}_{r=1}^{S!}`: 所有 `S!` 个置换矩阵

统一写法:

```text
X_t = [x_t^(1), x_t^(2), ..., x_t^(S)] in R^(S x d)

xhat_t = vec(X_t) in R^(S d)

xbar_t = RMSNorm(xhat_t)
```

当前实现里，三种方法都共享 mHC-lite 的基本流混合骨架:

1. 由当前多流状态生成 `H_pre`
2. 由当前多流状态生成 `H_res`
3. 用 `H_pre` 形成分支输入
4. 用 `H_res` 更新残差流
5. 将分支输出通过 `H_post` 写回多流

以下公式按当前新增三组配置的默认门控 `mhc_gate_fn="sigmoid"` 书写。若改成 `softmax`，则 `H_pre` 和 `H_post` 中的 `sigmoid` 改成对应维度的 `softmax`。

共享骨架可写成:

```text
g_pre_t = sigmoid(alpha_pre * xbar_t W_pre + b_pre)    in R^S

u_t = sum_{i=1}^S g_pre_t,i * x_t^(i)                  in R^d

X_t_tilde = (H_res_t)^T X_t                            in R^(S x d)

y_t = f_t(u_t)

g_post_t = 2 * sigmoid(alpha_post * xbar_t W_post + b_post)   in R^S

x_{t+1}^(j) = x_t_tilde^(j) + g_post_t,j * y_t,  j = 1, ..., S
```

三种方法的区别主要在于:

- `selective`: 如何构造 `H_res`
- `depth_attn`: 当前子层输入来自哪里
- `block_attn`: 用哪一组深度历史做注意力

---

## 方法一: Selective mHC-lite

### 核心思想

原始 mHC-lite 对所有 `S!` 个置换矩阵做 softmax 加权。`selective` 先做动态 top-k 选择，只保留当前 token 或位置最相关的 `k` 个 permutation，再在这个子集上做 softmax。

这样做有两个效果:

- `H_res` 仍然是置换矩阵的凸组合，因此仍然精确 doubly-stochastic
- 减少 permutation mixture 的平均化，增强选择性

### 算法流程图

```text
Input: multi-stream state X_t
  |
  v
Flatten + RMSNorm
  |
  v
Linear projection -> pre logits + permutation logits
  |
  +------------------------------+
  |                              |
  v                              v
pre logits                  permutation logits
  |                              |
  v                              v
sigmoid / softmax          Top-k selection
  |                              |
  v                              v
H_pre                     non-selected entries = -inf
  |                              |
  v                              v
branch input u_t          masked softmax
  |                              |
  v                              v
branch f_t                convex combination of permutations
  |                              |
  v                              v
branch output y_t               H_res
  |                              |
  +--------------+---------------+
                 |
                 v
      mix residual streams with H_res
                 |
                 v
         write y_t back with H_post
                 |
                 v
             output X_{t+1}
```

### 数学公式

先产生 permutation logits:

```text
z_t = alpha_res * xbar_t W_res + b_res      in R^(S!)
```

取前 `k` 个最大分量的索引集合:

```text
I_t = TopK(z_t, k)
```

构造 masked logits:

```text
z_t_tilde,r =
    z_t,r       if r in I_t
    -inf        if r not in I_t
```

然后在保留子集上做 softmax:

```text
pi_t,r = exp(z_t_tilde,r) / sum_{m=1}^{S!} exp(z_t_tilde,m)
```

构造残差混合矩阵:

```text
H_res_t = sum_{r=1}^{S!} pi_t,r * P_r
```

因为:

```text
pi_t,r >= 0
sum_{r=1}^{S!} pi_t,r = 1
```

且每个 `P_r` 都是置换矩阵，所以:

```text
H_res_t * 1 = 1
(H_res_t)^T * 1 = 1
```

因此 `H_res_t` 仍然在 Birkhoff polytope 中，保持精确 doubly-stochastic。

---

## 方法二: Depth-Attn mHC-lite

### 核心思想

这个方法把 Attention Residuals 的思想放到 mHC-lite 外层:

- mHC-lite 负责流维度的稳定混合
- depth attention 负责深度维度的选择性聚合

因此它同时针对两个问题:

- mHC-lite 关注的残差流稳定性
- Attention Residuals 关注的 PreNorm dilution

### 算法流程图

```text
Initial embedding / expanded streams v_0
  |
  v
History pool = {v_0}
  |
  v
For sublayer t:
    learnable query q_t
      |
      v
    depth softmax attention over history pool
      |
      v
    aggregated input x_t
      |
      v
    mHC-lite sublayer
      |
      v
    output v_t
      |
      v
    append v_t to history pool

After all sublayers:
    final query q_T
      |
      v
    attention over full history pool
      |
      v
    final output
```

### 数学公式

记第 `t` 个子层之前的历史为:

```text
V_{<t} = {v_0, v_1, ..., v_{t-1}}
```

其中:

- `v_0` 是初始 embedding 经 stream expansion 之后的状态
- `v_i` 是前面第 `i` 个子层经过 mHC-lite 后的输出

第 `t` 个子层有一个可学习 query:

```text
q_t in R^d
```

对每个历史状态做 RMSNorm 后计算深度注意力分数:

```text
e_{i->t} = q_t^T RMSNorm(v_i),    i < t
```

再做 softmax:

```text
a_{i->t} = exp(e_{i->t}) / sum_{j=0}^{t-1} exp(e_{j->t})
```

于是当前子层输入为:

```text
x_t = sum_{i=0}^{t-1} a_{i->t} * v_i
```

然后将其送入 mHC-lite:

```text
v_t = mHC-lite(x_t)
```

最终输出不是最后一个子层输出本身，而是再做一次最终聚合:

```text
h_out = sum_{i=0}^{T-1} a_{i->T}^{final} * v_i
```

其中:

```text
a_{i->T}^{final}
    = exp(q_T^T RMSNorm(v_i))
      / sum_{j=0}^{T-1} exp(q_T^T RMSNorm(v_j))
```

这里 `T` 是总子层数。在当前 GPT 实现里:

```text
T = 2L
```

因为每个 Transformer block 里有两个子层:

- attention 子层
- MLP 子层

---

## 方法三: Block-Attn mHC-lite

### 核心思想

`depth_attn` 对全部历史子层输出做 attention。`block_attn` 将深度历史压缩成:

- 初始状态
- 已完成 block 的 summary
- 当前 block 的 partial sum

这对应 Attention Residuals 中的 Block AttnRes 思路:

- 保留深度选择性
- 降低需要访问的深度状态数

### 算法流程图

```text
Initial state v_0
  |
  v
Start current block
  |
  v
Build source set:
    initial state
    completed block summaries
    current block partial sum (if exists)
  |
  v
Current sublayer query q_t
  |
  v
Depth softmax attention over source set
  |
  v
Aggregated input x_t
  |
  v
mHC-lite sublayer
  |
  v
Output v_t
  |
  v
Accumulate into current block partial sum
  |
  v
If block ends:
    freeze partial sum as block summary
Else:
    continue current block

After all blocks:
    final attention over final source set
      |
      v
    final output
```

### 数学公式

设 block 大小为 `G`。注意在当前实现里，`G` 的单位是“子层数”，不是 Transformer block 数。

令第 `n` 个 block 的子层索引集合为:

```text
B_n = {(n-1)G + 1, ..., nG}
```

第 `n` 个完整 block 的 summary 定义为:

```text
b_n = sum_{t in B_n} v_t
```

当前 block 中，到第 `t` 个子层之前的 partial sum 定义为:

```text
p_t = sum_{j=s_n}^{t-1} v_j
```

其中 `s_n` 是当前 block 的起始子层索引。

对第 `t` 个子层，source set 定义为:

```text
S_t = {v_0, b_1, ..., b_{n-1}}
```

如果当前 block 已经有 partial sum，则:

```text
S_t = S_t union {p_t}
```

设:

```text
S_t = {s_{t,1}, s_{t,2}, ..., s_{t,M_t}}
```

则深度注意力分数为:

```text
e_{m->t} = q_t^T RMSNorm(s_{t,m})
```

注意力权重为:

```text
a_{m->t} = exp(e_{m->t}) / sum_{r=1}^{M_t} exp(e_{r->t})
```

当前子层输入为:

```text
x_t = sum_{m=1}^{M_t} a_{m->t} * s_{t,m}
```

然后送入 mHC-lite:

```text
v_t = mHC-lite(x_t)
```

并更新当前 block 的 partial sum:

```text
p_{t+1} = p_t + v_t
```

当 block 结束时:

```text
b_n = p_{t+1}
```

最终输出同样通过最终 query 在最终 source set 上再聚合一次:

```text
h_out = sum_{m=1}^{M_final} a_m^{final} * s_m^{final}
```

---

## 三种方法的差异总结

### 1. `selective`

只改 `H_res` 的构造方式:

```text
all permutations softmax
    ->
Top-k + masked softmax
```

重点是更稀疏的 permutation routing，同时保留精确 doubly-stochastic。

### 2. `depth_attn`

只改子层输入来源:

```text
x_t = current residual state
    ->
x_t = depth-attended history state
```

重点是引入跨深度的内容相关选择。

### 3. `block_attn`

是 `depth_attn` 的低成本近似:

```text
history pool = all previous sublayer outputs
    ->
history pool = initial state + block summaries + current partial sum
```

重点是压缩深度历史，减少状态访问数量。

---

## 可直接写进论文方法部分的一句话概括

### Selective mHC-lite

我们在 mHC-lite 的 permutation mixture 上引入内容相关的 Top-k 稀疏选择，仅在少量候选置换矩阵上进行 softmax 归一化，从而在保持 `H_res` 精确 doubly-stochastic 的同时增强路由选择性。

### Depth-Attn mHC-lite

我们将 Attention Residuals 的深度 softmax 聚合与 mHC-lite 的流级稳定混合结合，在深度维进行内容相关的历史检索，在流维保持 Birkhoff 约束下的稳定残差传播。

### Block-Attn mHC-lite

我们进一步将全深度聚合压缩为 block 级聚合，仅对初始状态、已完成 block summary 与当前 block partial sum 做 depth attention，在保留主要深度选择性的同时降低历史状态开销。
