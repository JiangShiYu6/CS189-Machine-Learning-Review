# CS 189/289A - Discussion 10

## Fall 2025

---

> **Note:** Your TA will probably not cover all the problems on this worksheet. The discussion worksheets are not designed to be finished within an hour. They are deliberately made slightly longer so they can serve as resources you can use to practice, reinforce, and build upon concepts discussed in lectures, discussions, and homework.

## This week's cool AI demo/video

- [GPT 5.1 released today](https://openai.com/index/gpt-5-1/)
- [Unitree Fullbody teleop](https://www.youtube.com/watch?v=24h4FTH7plY)

---

## 1. Query-Key-Value in Transformer Attention

The attention mechanism was a key building block introduced by the paper *Attention Is All You Need* in 2017 that has jump-started unprecedented advances in deep learning architectures. Attention helps machine learning models determine the relative importance of each part of an input sequence to other parts of the input sequence. In this problem, we will see how queries, keys, and values are calculated in the attention process and how they allow models to attend to their inputs.

Assume that the encoder in our transformer is processing the embeddings of three tokens:

$$
\mathbf{x}_1=
\begin{bmatrix}
2\\
3
\end{bmatrix},
\qquad
\mathbf{x}_2=
\begin{bmatrix}
-1\\
0
\end{bmatrix},
\qquad
\mathbf{x}_3=
\begin{bmatrix}
1\\
-2
\end{bmatrix}.
$$

Our attention layer has the following weight matrices:

$$
\mathbf{W}_Q=
\begin{bmatrix}
1 & 0 & 2\\
0 & 1 & -3
\end{bmatrix},
\qquad
\mathbf{W}_K=
\begin{bmatrix}
0 & 0 & -1\\
-2 & 0 & 1
\end{bmatrix},
\qquad
\mathbf{W}_V=
\begin{bmatrix}
8 & 0\\
0 & 9
\end{bmatrix}.
$$

The query, key, and value of each embedding vector are defined respectively as

$$
\mathbf{q}_i=\mathbf{W}_Q^\top\mathbf{x}_i,
\qquad
\mathbf{k}_i=\mathbf{W}_K^\top\mathbf{x}_i,
\qquad
\mathbf{v}_i=\mathbf{W}_V^\top\mathbf{x}_i.
$$

### (a)

Compute the query, key, and value vectors for $\mathbf{x}_1$, $\mathbf{x}_2$, and $\mathbf{x}_3$.

**Answer (a):**

First transpose the three weight matrices:

$$
\mathbf{W}_Q^\top=
\begin{bmatrix}
1&0\\
0&1\\
2&-3
\end{bmatrix},
\qquad
\mathbf{W}_K^\top=
\begin{bmatrix}
0&-2\\
0&0\\
-1&1
\end{bmatrix},
\qquad
\mathbf{W}_V^\top=
\begin{bmatrix}
8&0\\
0&9
\end{bmatrix}.
$$

For $\mathbf{x}_1=[2,3]^\top$,

$$
\mathbf{q}_1=\mathbf{W}_Q^\top\mathbf{x}_1
=\begin{bmatrix}2\\3\\-5\end{bmatrix},
\qquad
\mathbf{k}_1=\mathbf{W}_K^\top\mathbf{x}_1
=\begin{bmatrix}-6\\0\\1\end{bmatrix},
\qquad
\mathbf{v}_1=\mathbf{W}_V^\top\mathbf{x}_1
=\begin{bmatrix}16\\27\end{bmatrix}.
$$

For $\mathbf{x}_2=[-1,0]^\top$,

$$
\mathbf{q}_2=\mathbf{W}_Q^\top\mathbf{x}_2
=\begin{bmatrix}-1\\0\\-2\end{bmatrix},
\qquad
\mathbf{k}_2=\mathbf{W}_K^\top\mathbf{x}_2
=\begin{bmatrix}0\\0\\1\end{bmatrix},
\qquad
\mathbf{v}_2=\mathbf{W}_V^\top\mathbf{x}_2
=\begin{bmatrix}-8\\0\end{bmatrix}.
$$

For $\mathbf{x}_3=[1,-2]^\top$,

$$
\mathbf{q}_3=\mathbf{W}_Q^\top\mathbf{x}_3
=\begin{bmatrix}1\\-2\\8\end{bmatrix},
\qquad
\mathbf{k}_3=\mathbf{W}_K^\top\mathbf{x}_3
=\begin{bmatrix}4\\0\\-3\end{bmatrix},
\qquad
\mathbf{v}_3=\mathbf{W}_V^\top\mathbf{x}_3
=\begin{bmatrix}8\\-18\end{bmatrix}.
$$


### (b)

Recall that the attention mechanism in transformers allows the model to decide how much each token should "focus" on every other token in the sequence. To do this, the model computes an attention score for every ordered pair of tokens by first taking the dot product between the query vector of one token and the key vector of another. For example, the inner product between the $i$th token's query, $\mathbf{q}_i$, and the $j$th token's key, $\mathbf{k}_j$, is

$$
z_{i,j}=\mathbf{q}_i^\top\mathbf{k}_j.
$$

The inner product $z_{i,j}$ measures how token $\mathbf{x}_i$ should consider or "pay attention" to token $\mathbf{x}_j$. Calculate the attention that token $\mathbf{x}_3$ places on each of the three tokens $\mathbf{x}_1$, $\mathbf{x}_2$, and $\mathbf{x}_3$.

**Answer (b):**

Because we are computing how token $\mathbf{x}_3$ attends to the other tokens, we use the query vector $\mathbf{q}_3$ and take its dot product with each key vector:

$$
z_{3,j}=\mathbf{q}_3^\top\mathbf{k}_j.
$$

Using the vectors from part (a),

$$
z_{3,1}=\mathbf{q}_3^\top\mathbf{k}_1
=
\begin{bmatrix}1&-2&8\end{bmatrix}
\begin{bmatrix}-6\\0\\1\end{bmatrix}
=1(-6)+(-2)(0)+8(1)
=\boxed{2},
$$

$$
z_{3,2}=\mathbf{q}_3^\top\mathbf{k}_2
=
\begin{bmatrix}1&-2&8\end{bmatrix}
\begin{bmatrix}0\\0\\1\end{bmatrix}
=1(0)+(-2)(0)+8(1)
=\boxed{8},
$$

and

$$
z_{3,3}=\mathbf{q}_3^\top\mathbf{k}_3
=
\begin{bmatrix}1&-2&8\end{bmatrix}
\begin{bmatrix}4\\0\\-3\end{bmatrix}
=1(4)+(-2)(0)+8(-3)
=\boxed{-20}.
$$

Therefore, the attention scores of token $\mathbf{x}_3$ for $\mathbf{x}_1$, $\mathbf{x}_2$, and $\mathbf{x}_3$ are

$$
\boxed{(z_{3,1},z_{3,2},z_{3,3})=(2,8,-20)}.
$$


### (c)

We would like to transform these attention scores into probabilities, so we will apply the softmax function. Taking the softmax over $z_{3,1}$, $z_{3,2}$, and $z_{3,3}$, we obtain

$$
a_{3,1}\approx0.00247,
\qquad
a_{3,2}\approx0.99753,
\qquad
a_{3,3}\approx6.90\times10^{-13}.
$$

The resulting softmax scores act as weights for forming a weighted sum of the value vectors. Write an expression for this weighted sum for $\mathbf{x}_3$ and plug in the values you computed previously.

**Answer (c):**

The softmax scores are used as weights for the corresponding value vectors. Thus, the attention output for token $\mathbf{x}_3$ is

$$
\operatorname{output}_3
=a_{3,1}\mathbf{v}_1+a_{3,2}\mathbf{v}_2+a_{3,3}\mathbf{v}_3.
$$

Substituting the probabilities and the value vectors from part (a),

$$
\begin{aligned}
\operatorname{output}_3
&\approx
0.00247
\begin{bmatrix}16\\27\end{bmatrix}
+0.99753
\begin{bmatrix}-8\\0\end{bmatrix}
+\left(6.90\times10^{-13}\right)
\begin{bmatrix}8\\-18\end{bmatrix}\\
&\approx
\boxed{
\begin{bmatrix}
-7.94072\\
0.06669
\end{bmatrix}}.
\end{aligned}
$$

Because $a_{3,2}\approx0.99753$, the output is very close to $\mathbf{v}_2=[-8,0]^\top$.


### (d)

Transformers benefit from efficient matrix operations. To parallelize our computations, we stack all our input embeddings, $\mathbf{x}_1$, $\mathbf{x}_2$, and $\mathbf{x}_3$, into the rows of a data matrix $\mathbf{X}$:

$$
\mathbf{X}=
\begin{bmatrix}
\mathbf{x}_1^\top\\
\mathbf{x}_2^\top\\
\mathbf{x}_3^\top
\end{bmatrix}
=
\begin{bmatrix}
2 & 3\\
-1 & 0\\
1 & -2
\end{bmatrix}.
$$

We want to obtain query, key, and value matrices, where

$$
\mathbf{Q}=
\begin{bmatrix}
\mathbf{q}_1^\top\\
\mathbf{q}_2^\top\\
\mathbf{q}_3^\top
\end{bmatrix},
\qquad
\mathbf{K}=
\begin{bmatrix}
\mathbf{k}_1^\top\\
\mathbf{k}_2^\top\\
\mathbf{k}_3^\top
\end{bmatrix},
\qquad
\mathbf{V}=
\begin{bmatrix}
\mathbf{v}_1^\top\\
\mathbf{v}_2^\top\\
\mathbf{v}_3^\top
\end{bmatrix}.
$$

Write an expression for these three matrices in terms of the data matrix $\mathbf{X}$ and the weight matrices defined in the problem statement. What are the dimensions of these matrices?

**Answer (d):**

Because each row of $\mathbf{X}$ is an input vector transposed, the query, key, and value matrices are

$$
\boxed{
\mathbf{Q}=\mathbf{X}\mathbf{W}_Q,
\qquad
\mathbf{K}=\mathbf{X}\mathbf{W}_K,
\qquad
\mathbf{V}=\mathbf{X}\mathbf{W}_V}.
$$

In particular,

$$
\mathbf{Q}
=
\begin{bmatrix}
2&3\\
-1&0\\
1&-2
\end{bmatrix}
\begin{bmatrix}
1&0&2\\
0&1&-3
\end{bmatrix}
=
\begin{bmatrix}
2&3&-5\\
-1&0&-2\\
1&-2&8
\end{bmatrix},
$$

$$
\mathbf{K}
=
\begin{bmatrix}
2&3\\
-1&0\\
1&-2
\end{bmatrix}
\begin{bmatrix}
0&0&-1\\
-2&0&1
\end{bmatrix}
=
\begin{bmatrix}
-6&0&1\\
0&0&1\\
4&0&-3
\end{bmatrix},
$$

and

$$
\mathbf{V}
=
\begin{bmatrix}
2&3\\
-1&0\\
1&-2
\end{bmatrix}
\begin{bmatrix}
8&0\\
0&9
\end{bmatrix}
=
\begin{bmatrix}
16&27\\
-8&0\\
8&-18
\end{bmatrix}.
$$

Since $\mathbf{X}\in\mathbb{R}^{3\times2}$, $\mathbf{W}_Q,\mathbf{W}_K\in\mathbb{R}^{2\times3}$, and $\mathbf{W}_V\in\mathbb{R}^{2\times2}$, their dimensions are

$$
\boxed{
\mathbf{Q}\in\mathbb{R}^{3\times3},
\qquad
\mathbf{K}\in\mathbb{R}^{3\times3},
\qquad
\mathbf{V}\in\mathbb{R}^{3\times2}}.
$$


### (e)

Using the $\mathbf{Q}$ and $\mathbf{K}$ matrices from the previous step, show that the $(i,j)$th entry of the matrix product $\mathbf{Q}\mathbf{K}^\top$ is exactly $z_{i,j}$ from part (b).

**Answer (e):**

By construction, the $i$th row of $\mathbf{Q}$ is $\mathbf{q}_i^\top$. Similarly, since the $j$th row of $\mathbf{K}$ is $\mathbf{k}_j^\top$, the $j$th column of $\mathbf{K}^\top$ is $\mathbf{k}_j$. Therefore, the $(i,j)$th entry of the matrix product is

$$
\begin{aligned}
(\mathbf{Q}\mathbf{K}^\top)_{i,j}
&=(\text{$i$th row of }\mathbf{Q})(\text{$j$th column of }\mathbf{K}^\top)\\
&=\mathbf{q}_i^\top\mathbf{k}_j\\
&=\boxed{z_{i,j}}.
\end{aligned}
$$

Thus,

$$
\mathbf{Q}\mathbf{K}^\top
=
\begin{bmatrix}
z_{1,1}&z_{1,2}&z_{1,3}\\
z_{2,1}&z_{2,2}&z_{2,3}\\
z_{3,1}&z_{3,2}&z_{3,3}
\end{bmatrix}.
$$

For example, its $(3,2)$ entry is

$$
(\mathbf{Q}\mathbf{K}^\top)_{3,2}
=\mathbf{q}_3^\top\mathbf{k}_2
=z_{3,2}
=8,
$$

which agrees with part (b).


### (f)

The $i$th row in the matrix product $\mathbf{Q}\mathbf{K}^\top$ represents how much $\mathbf{x}_i$ should attend to every other token $\mathbf{x}_j$. Recall that we apply the softmax over the attention scores $z_{i,1}$, $z_{i,2}$, and $z_{i,3}$ to turn them into probabilities. In matrix form, this means we apply the softmax to each row of $\mathbf{Q}\mathbf{K}^\top$ so that all entries are non-negative and the entries in each row sum to $1$.

Let $\mathbf{A}$ be the result of applying the softmax function to $\mathbf{Q}\mathbf{K}^\top$ row-wise. Show that the $i$th row of $\mathbf{A}\mathbf{V}$ is the weighted sum of the value vectors for $\mathbf{x}_i$, using the softmax scores as weights.

**Answer (f):**

Let $a_{i,j}$ denote the $(i,j)$th entry of $\mathbf{A}$. Since $\mathbf{A}$ is obtained by applying softmax to each row of $\mathbf{Q}\mathbf{K}^\top$,

$$
a_{i,j}
=\frac{\exp(z_{i,j})}{\sum_{\ell=1}^{3}\exp(z_{i,\ell})},
\qquad
a_{i,j}\geq 0,
\qquad
\sum_{j=1}^{3}a_{i,j}=1.
$$

The $i$th row of $\mathbf{A}$ is $[a_{i,1},a_{i,2},a_{i,3}]$, while the rows of $\mathbf{V}$ are the transposed value vectors. Therefore, by the definition of matrix multiplication,

$$
\begin{aligned}
(\mathbf{A}\mathbf{V})_{i,:}
&=
\begin{bmatrix}
a_{i,1}&a_{i,2}&a_{i,3}
\end{bmatrix}
\begin{bmatrix}
\mathbf{v}_1^\top\\
\mathbf{v}_2^\top\\
\mathbf{v}_3^\top
\end{bmatrix}\\
&=\boxed{a_{i,1}\mathbf{v}_1^\top
+a_{i,2}\mathbf{v}_2^\top
+a_{i,3}\mathbf{v}_3^\top}\\
&=\boxed{\sum_{j=1}^{3}a_{i,j}\mathbf{v}_j^\top}.
\end{aligned}
$$

Hence, the $i$th row of $\mathbf{A}\mathbf{V}$ is exactly the weighted sum of all value vectors for token $\mathbf{x}_i$, with its row-wise softmax attention scores as the weights.

---

## 2. Justifying Scaled-Dot Product Attention

In the previous problem, we worked through an example of softmax inner-product self-attention. In transformers, we apply scaled softmax inner-product self-attention. In this problem, we will explore where this scaling factor comes from.

Suppose $\mathbf{q},\mathbf{k}\in\mathbb{R}^{D_k}$ are two random vectors with

$$
\mathbf{q},\mathbf{k}\overset{\text{iid}}{\sim}
\mathcal{N}(\mu\mathbf{1},\sigma^2\mathbf{I}),
$$

where $\mu\mathbf{1}\in\mathbb{R}^{D_k}$ and $\sigma\in\mathbb{R}^+$. In other words, each component $q_i$ of $\mathbf{q}$ is drawn from a normal distribution with mean $\mu$ and standard deviation $\sigma$, and the same is true for $k_i$ of $\mathbf{k}$.

### (a)

Define $\mathbb{E}[\mathbf{q}^\top\mathbf{k}]$ in terms of $\mu$, $\sigma$, and $D_k$.

**Answer (a):**

Using the definition of the dot product and linearity of expectation,

$$
\mathbb{E}[\mathbf{q}^\top\mathbf{k}]
=\mathbb{E}\left[\sum_{i=1}^{D_k}q_i k_i\right]
=\sum_{i=1}^{D_k}\mathbb{E}[q_i k_i].
$$

Because $\mathbf{q}$ and $\mathbf{k}$ are independent, $q_i$ and $k_i$ are independent. Thus,

$$
\mathbb{E}[q_i k_i]
=\mathbb{E}[q_i]\mathbb{E}[k_i]
=\mu^2.
$$

Therefore,

$$
\boxed{\mathbb{E}[\mathbf{q}^\top\mathbf{k}]=D_k\mu^2}.
$$

The result does not depend on $\sigma$ because independence makes the expectation of each product equal to the product of the two means.

### (b)

Considering a practical case where $\mu=0$ and $\sigma=1$, define $\operatorname{Var}(\mathbf{q}^\top\mathbf{k})$ in terms of $D_k$.

**Answer (b):**

When $\mu=0$ and $\sigma=1$, each $q_i$ and $k_i$ is an independent standard normal random variable. Since the products $q_i k_i$ are independent across dimensions,

$$
\operatorname{Var}(\mathbf{q}^\top\mathbf{k})
=\operatorname{Var}\left(\sum_{i=1}^{D_k}q_i k_i\right)
=\sum_{i=1}^{D_k}\operatorname{Var}(q_i k_i).
$$

For each dimension,

$$
\begin{aligned}
\operatorname{Var}(q_i k_i)
&=\mathbb{E}[q_i^2k_i^2]
-\bigl(\mathbb{E}[q_i k_i]\bigr)^2\\
&=\mathbb{E}[q_i^2]\mathbb{E}[k_i^2]-0\\
&=1\cdot1=1.
\end{aligned}
$$

Here $\mathbb{E}[q_i^2]=\mathbb{E}[k_i^2]=1$ because a standard normal variable has mean $0$ and variance $1$, and $\mathbb{E}[X^2]=\operatorname{Var}(X)+(\mathbb{E}[X])^2$.

Consequently,

$$
\boxed{\operatorname{Var}(\mathbf{q}^\top\mathbf{k})=D_k}.
$$

### (c)

Continue to assume $\mu=0$ and $\sigma=1$. Let $s$ be the scaling factor on the dot product. Suppose we want

$$
\mathbb{E}\left[\frac{\mathbf{q}^\top\mathbf{k}}{s}\right]=0,
\qquad
\operatorname{Var}\left(\frac{\mathbf{q}^\top\mathbf{k}}{s}\right)=1.
$$

What should $s$ be in terms of $D_k$?

**Answer (c):**

From the previous parts,

$$
\mathbb{E}[\mathbf{q}^\top\mathbf{k}]=0,
\qquad
\operatorname{Var}(\mathbf{q}^\top\mathbf{k})=D_k.
$$

Scaling by $1/s$ leaves the expectation equal to zero:

$$
\mathbb{E}\left[\frac{\mathbf{q}^\top\mathbf{k}}{s}\right]
=\frac{1}{s}\mathbb{E}[\mathbf{q}^\top\mathbf{k}]
=0.
$$

The variance is

$$
\operatorname{Var}\left(\frac{\mathbf{q}^\top\mathbf{k}}{s}\right)
=\frac{1}{s^2}\operatorname{Var}(\mathbf{q}^\top\mathbf{k})
=\frac{D_k}{s^2}.
$$

Requiring this variance to equal $1$ gives

$$
\frac{D_k}{s^2}=1
\quad\Longrightarrow\quad
s^2=D_k.
$$

Since the scaling factor is positive,

$$
\boxed{s=\sqrt{D_k}}.
$$

Thus scaled dot-product attention uses $\mathbf{q}^\top\mathbf{k}/\sqrt{D_k}$, whose variance remains $1$ instead of growing with the key dimension.
