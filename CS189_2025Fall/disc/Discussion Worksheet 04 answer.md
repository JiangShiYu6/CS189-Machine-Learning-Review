# CS 189/289A - Discussion 4

## Fall 2025

---

> **Note:** Your TA will probably not cover all the problems on this worksheet. The discussion worksheets are not designed to be finished within an hour. They are deliberately made slightly longer so they can serve as resources you can use to practice, reinforce, and build upon concepts discussed in lectures, discussions, and homework.

## This week's cool AI demo/video

[Oasis starting point](https://oasis.decart.ai/starting-point)

---

## 1. Gaussian Mixture Clustering with Kangaroos and Berkeley Students (1D, Two Gaussians)

Consider a Gaussian Mixture Model (GMM) with two components in one dimension. Let

$$
z_n \in \{K,B\}
$$

be the latent (unobserved) class of the $n$-th jumper ($K$ = Kangaroo and $B$ = Berkeley student), and let

$$
x_n \in \mathbb{R}
$$

be the observed jump height (in meters).

Assume that the classes occur with equal probability and that, conditional on the class, jump heights are Gaussian with class-dependent means $\mu_K$ and $\mu_B$ (the average jump heights of Kangaroos and Berkeley students, respectively), and a shared variance $\sigma^2$ (the variability in jump heights).

### (a) Write down the following quantities explicitly

#### (i) Prior

The prior probability that a jumper is a Kangaroo or Berkeley student before observing the jump height:

$$
p(z_n).
$$

**Answer (a)(i):** Since the two classes have equal prior probability,

$$
p(z_n=K)=p(z_n=B)=\frac{1}{2}.
$$

#### (ii) Likelihood

The likelihood of the jump height conditioned on whether the jumper is a Kangaroo or Berkeley student:

$$
p(x_n \mid z_n).
$$

**Answer (a)(ii):** For either class $k\in\{K,B\}$,

$$
p(x_n\mid z_n=k)
=\mathcal{N}(x_n;\mu_k,\sigma^2)
=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x_n-\mu_k)^2}{2\sigma^2}\right).
$$

#### (iii) Marginal

The marginal probability that a certain jump height is observed:

$$
p(x_n).
$$

**Answer (a)(iii):** Marginalizing over the unknown class gives

$$
\begin{aligned}
p(x_n)
&=\sum_{k\in\{K,B\}}p(x_n\mid z_n=k)p(z_n=k)\\
&=\frac{1}{2}\mathcal{N}(x_n;\mu_K,\sigma^2)
+\frac{1}{2}\mathcal{N}(x_n;\mu_B,\sigma^2).
\end{aligned}
$$

#### (iv) Posterior

The posterior probability of a jumper being a Kangaroo or Berkeley student after observing the jump height:

$$
p(z_n \mid x_n).
$$

**Answer (a)(iv):** By Bayes' rule, for $k\in\{K,B\}$,

$$
\begin{aligned}
p(z_n=k\mid x_n)
&=\frac{p(x_n\mid z_n=k)p(z_n=k)}{p(x_n)}\\
&=\frac{\mathcal{N}(x_n;\mu_k,\sigma^2)}
{\mathcal{N}(x_n;\mu_K,\sigma^2)
+\mathcal{N}(x_n;\mu_B,\sigma^2)}.
\end{aligned}
$$

The factors of $1/2$ cancel because the two classes have equal prior probabilities.

---

### (b) Soft assignments and the expected log-likelihood

Recall the K-means objective:

$$
J(\{r_{nk}\},\{\mu_k\})
=
\sum_{n=1}^{N}\sum_{k=1}^{K}
r_{nk}\lVert x_n-\mu_k\rVert^2,
$$

where $\mu_k$ is the mean of cluster $k$ and $r_{nk}\in\{0,1\}$ is a binary indicator variable that describes which of the $K$ clusters the data point $x_n$ is assigned to. The K-means algorithm alternates between updating the cluster assignments and updating the cluster centers.

We modify this algorithm to generate GMMs in the following way:

1. **Update assignments:** Replace the hard assignments $r_{nk}$ with soft assignments $\gamma_{nk}$, representing the probability that a point $x_n$ belongs to cluster $k$.

2. **Update parameters:** Rather than selecting cluster centers that minimize the squared distances of data points to their assigned cluster centers, select the parameters $\theta$ that maximize the $Q$ function:

$$
Q(\theta)
=
\sum_{n=1}^{N}\sum_{k=1}^{K}
\gamma_{nk}\ln p(x_n,z_n=k\mid\theta).
$$

#### (i)

Identify which probability from part (a) corresponds to $\gamma_{nk}$. Verify that

$$
\sum_k \gamma_{nk}=1.
$$

**Answer (b)(i):** The soft assignment is the posterior probability from part (a)(iv):

$$
\gamma_{nk}=p(z_n=k\mid x_n).
$$

Since $z_n$ must belong to exactly one of the classes,

$$
\sum_{k\in\{K,B\}}\gamma_{nk}
=\sum_{k\in\{K,B\}}p(z_n=k\mid x_n)
=1.
$$

#### (ii)

Show that the $Q$ function is the expected log-likelihood of the observed data

$$
X=(x_1,\ldots,x_N)
$$

and the unobserved latent vector

$$
Z=(z_1,\ldots,z_N)
$$

under the posterior probability of $Z$.[^1]

**Answer (b)(ii):** Substituting
$\gamma_{nk}=p(z_n=k\mid x_n)$ into the definition of $Q$ gives

$$
\begin{aligned}
Q(\theta)
&=\sum_{n=1}^{N}\sum_{k=1}^{K}
p(z_n=k\mid x_n)\ln p(x_n,z_n=k\mid\theta)\\
&=\sum_{n=1}^{N}
\mathbb{E}_{z_n\sim p(z_n\mid x_n)}
\left[\ln p(x_n,z_n\mid\theta)\right].
\end{aligned}
$$

Because the data points and their latent assignments are independent across $n$,

$$
p(X,Z\mid\theta)=\prod_{n=1}^{N}p(x_n,z_n\mid\theta).
$$

Therefore,

$$
\begin{aligned}
Q(\theta)
&=\mathbb{E}_{Z\sim p(Z\mid X)}
\left[\sum_{n=1}^{N}\ln p(x_n,z_n\mid\theta)\right]\\
&=\mathbb{E}_{Z\sim p(Z\mid X)}
\left[\ln p(X,Z\mid\theta)\right].
\end{aligned}
$$

Thus, $Q(\theta)$ is the posterior expectation of the complete-data log-likelihood.

#### (iii)

Derive the maximum likelihood estimate for $\mu_k$ using the expected log-likelihood $Q(\theta)$, treating $\gamma_{nk}$ as constants.

**Answer (b)(iii):** For a fixed component $k$, the terms in $Q(\theta)$ that depend on $\mu_k$ are

$$
\begin{aligned}
Q_k(\mu_k)
&=\sum_{n=1}^{N}\gamma_{nk}
\ln\mathcal{N}(x_n;\mu_k,\sigma^2)+\text{constant}\\
&=-\frac{1}{2\sigma^2}
\sum_{n=1}^{N}\gamma_{nk}(x_n-\mu_k)^2
+\text{constant}.
\end{aligned}
$$

Differentiate with respect to $\mu_k$:

$$
\frac{\partial Q}{\partial\mu_k}
=\frac{1}{\sigma^2}
\sum_{n=1}^{N}\gamma_{nk}(x_n-\mu_k).
$$

Setting the derivative equal to zero gives

$$
\sum_{n=1}^{N}\gamma_{nk}x_n
-\mu_k\sum_{n=1}^{N}\gamma_{nk}=0.
$$

Hence,

$$
\boxed{
\widehat{\mu}_k
=\frac{\sum_{n=1}^{N}\gamma_{nk}x_n}
{\sum_{n=1}^{N}\gamma_{nk}}
}.
$$

Moreover,

$$
\frac{\partial^2Q}{\partial\mu_k^2}
=-\frac{1}{\sigma^2}\sum_{n=1}^{N}\gamma_{nk}<0
$$

whenever the component has positive total responsibility. Therefore, this stationary point is a maximum. The estimate is the responsibility-weighted mean of the observations assigned to component $k$.

[^1]: We optimize the joint likelihood because it is much more tractable to compute while still optimizing the marginal likelihood. See [Expectation-maximization algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) for more information.

---

## 2. Closed-Form Solution for $\ell_2$-Regularized Least Squares (Ridge)

Consider a data matrix

$$
X\in\mathbb{R}^{N\times(D+1)}
$$

with rows $x_n^\top$, a target vector $t\in\mathbb{R}^N$, and a parameter vector $w\in\mathbb{R}^{D+1}$. For a regularization hyperparameter $\lambda\ge0$, define the ridge objective:

$$
E(w)=\lVert Xw-t\rVert_2^2+\lambda\lVert w\rVert_2^2.
$$

### (a)

Write $E(w)$ in expanded quadratic form

$$
w^\top Aw-2b^\top w+c
$$

by identifying $A$, $b$, and $c$ in terms of $X$, $t$, and $\lambda$.

**Answer (a):** Expand the squared-error term:

$$
\begin{aligned}
E(w)
&=(Xw-t)^\top(Xw-t)+\lambda w^\top w\\
&=w^\top X^\top Xw-w^\top X^\top t-t^\top Xw
+t^\top t+\lambda w^\top w.
\end{aligned}
$$

Because $w^\top X^\top t$ and $t^\top Xw$ are equal scalars, the two cross terms can be combined:

$$
E(w)
=w^\top(X^\top X+\lambda I)w
-2t^\top Xw+t^\top t.
$$

Comparing this expression with
$w^\top Aw-2b^\top w+c$, we obtain

$$
\boxed{
A=X^\top X+\lambda I,
\qquad
b=X^\top t,
\qquad
c=t^\top t
}.
$$

In particular, $b^\top=t^\top X$, so $b^\top w=t^\top Xw$.

### (b)

Compute the gradient $\nabla_w E(w)$ using the identities

$$
\nabla_w\lVert Xw-t\rVert_2^2
=2X^\top(Xw-t),
\qquad
\nabla_w\lVert w\rVert_2^2=2w.
$$

**Answer (b):** Applying the given gradient identities term by term,

$$
\begin{aligned}
\nabla_w E(w)
&=\nabla_w\lVert Xw-t\rVert_2^2
+\lambda\nabla_w\lVert w\rVert_2^2\\
&=2X^\top(Xw-t)+2\lambda w.
\end{aligned}
$$

Equivalently, after expanding and collecting the terms involving $w$,

$$
\boxed{
\nabla_w E(w)
=2(X^\top X+\lambda I)w-2X^\top t
}.
$$

### (c)

Set the gradient to zero and derive the normal equations for ridge regression. Solve for $w$.

**Answer (c):** Set the gradient from part (b) equal to zero:

$$
2X^\top(Xw-t)+2\lambda w=0.
$$

Dividing by $2$ and collecting the terms involving $w$ gives

$$
X^\top Xw-X^\top t+\lambda w=0,
$$

so the ridge normal equations are

$$
\boxed{(X^\top X+\lambda I)w=X^\top t}.
$$

When $X^\top X+\lambda I$ is invertible, the solution is

$$
\boxed{
\widehat{w}=(X^\top X+\lambda I)^{-1}X^\top t
}.
$$

### (d)

Justify why $X^\top X+\lambda I$ is invertible for $\lambda>0$, and discuss when it might fail for $\lambda=0$.

**Answer (d):** For any nonzero vector $v\in\mathbb{R}^{D+1}$,

$$
\begin{aligned}
v^\top(X^\top X+\lambda I)v
&=v^\top X^\top Xv+\lambda v^\top v\\
&=\lVert Xv\rVert_2^2+\lambda\lVert v\rVert_2^2.
\end{aligned}
$$

When $\lambda>0$, the second term is strictly positive for every $v\ne0$, while the first term is nonnegative. Therefore,

$$
v^\top(X^\top X+\lambda I)v>0,
$$

so $X^\top X+\lambda I$ is positive definite and hence invertible.

When $\lambda=0$, the matrix reduces to $X^\top X$. It is invertible if and only if the columns of $X$ are linearly independent, equivalently,

$$
\operatorname{rank}(X)=D+1.
$$

Thus, invertibility can fail when $X$ is rank deficient—for example, when some features are linearly dependent or when there are more features than observations. In that case, ordinary least squares may have multiple minimizing solutions, whereas a positive ridge parameter guarantees a unique solution.
