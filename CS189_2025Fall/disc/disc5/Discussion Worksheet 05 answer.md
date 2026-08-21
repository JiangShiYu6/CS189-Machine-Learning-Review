# CS 189/289A - Discussion 5

## Fall 2025

---

> **Note:** Your TA will probably not cover all the problems on this worksheet. The discussion worksheets are not designed to be finished within an hour. They are deliberately made slightly longer so they can serve as resources you can use to practice, reinforce, and build upon concepts discussed in lectures, discussions, and homework.

## This week's cool AI demo/video

[Deep Dream](https://www.youtube.com/watch?v=DgPaCWJL7XI)

[Explanation for Deep Dream](https://youtu.be/ta5fdaqDT3M?si=TfGM_QnBBHoJ4sbl&t=2590)

---

## 1. Maximum Likelihood Estimation with Laplace Noise

Suppose we have a regression model with parameters $w\in\mathbb{R}^D$ and observations $(x_n,t_n)$ for $n=1,\ldots,N$. We assume that the observed targets are generated as

$$
t_n=x_n^\top w+\epsilon_n,
$$

where the noise terms $\epsilon_n$ are i.i.d. from a Laplace distribution:

$$
p(\epsilon)=\frac{1}{2b}\exp\left(-\frac{|\epsilon|}{b}\right).
$$

### (a)

Write down the likelihood $p(\mathcal{D}\mid w)$ for the parameters $w$ given the dataset

$$
\mathcal{D}=\{(x_n,t_n)\}_{n=1}^{N}.
$$

**Answer (a):** Define the residual for observation $n$ as

$$
\epsilon_n=t_n-x_n^\top w.
$$

Since $\epsilon_n\sim\operatorname{Laplace}(0,b)$, shifting the Laplace density by $x_n^\top w$ gives the conditional density of one target:

$$
\begin{aligned}
t_n\mid x_n,w
&\sim\operatorname{Laplace}(x_n^\top w,b),\\
p(t_n\mid x_n,w)
&=\frac{1}{2b}
\exp\left(-\frac{|t_n-x_n^\top w|}{b}\right).
\end{aligned}
$$

The noise variables are i.i.d., so the targets are conditionally independent given $X$ and $w$. Therefore, the likelihood is the product of the individual conditional densities:

$$
\begin{aligned}
p(\mathcal{D}\mid w)
&=\prod_{n=1}^{N}p(t_n\mid x_n,w)\\
&=\prod_{n=1}^{N}
\frac{1}{2b}
\exp\left(-\frac{|t_n-x_n^\top w|}{b}\right).
\end{aligned}
$$

Equivalently, the constants and exponential terms can be combined:

$$
\boxed{
p(\mathcal{D}\mid w)
=(2b)^{-N}
\exp\left(-\frac{1}{b}
\sum_{n=1}^{N}|t_n-x_n^\top w|\right)
}.
$$

### (b)

What is the log-likelihood? Simplify your answer as much as possible.

**Answer (b):** Taking the logarithm converts the product in part (a) into a sum:

$$
\begin{aligned}
\log p(\mathcal{D}\mid w)
&=\sum_{n=1}^{N}
\log\left[
\frac{1}{2b}
\exp\left(-\frac{|t_n-x_n^\top w|}{b}\right)
\right]\\
&=\sum_{n=1}^{N}
\left[
\log\left(\frac{1}{2b}\right)
-\frac{|t_n-x_n^\top w|}{b}
\right]\\
&=-N\log(2b)
-\frac{1}{b}\sum_{n=1}^{N}|t_n-x_n^\top w|.
\end{aligned}
$$

Hence,

$$
\boxed{
\log p(\mathcal{D}\mid w)
=-N\log(2b)
-\frac{1}{b}\sum_{n=1}^{N}|t_n-x_n^\top w|
}.
$$

### (c)

Show that maximizing the log-likelihood is equivalent to minimizing the **mean absolute error (MAE)**, defined as

$$
\operatorname{MAE}(w)
=\frac{1}{N}\sum_{n=1}^{N}|t_n-x_n^\top w|.
$$

**Answer (c):** From part (b), the maximum likelihood estimate is

$$
\widehat{w}_{\mathrm{MLE}}
=\operatorname*{arg\,max}_{w}
\left[
-N\log(2b)
-\frac{1}{b}\sum_{n=1}^{N}|t_n-x_n^\top w|
\right].
$$

The term $-N\log(2b)$ does not depend on $w$, so it has no effect on the optimizer. Since $b>0$, maximizing the remaining negative term is equivalent to minimizing the sum of absolute residuals:

$$
\widehat{w}_{\mathrm{MLE}}
=\operatorname*{arg\,min}_{w}
\sum_{n=1}^{N}|t_n-x_n^\top w|.
$$

Multiplying an objective by the positive constant $1/N$ also leaves its minimizer unchanged. Thus,

$$
\begin{aligned}
\widehat{w}_{\mathrm{MLE}}
&=\operatorname*{arg\,min}_{w}
\frac{1}{N}\sum_{n=1}^{N}|t_n-x_n^\top w|\\
&=\boxed{
\operatorname*{arg\,min}_{w}\operatorname{MAE}(w)
}.
\end{aligned}
$$

The absolute value appears because the assumed noise distribution is Laplace. This objective is also called the Laplace loss.

### (d)

Now consider instead a standard regression model with Gaussian noise. Let

$$
t=(t_1,\ldots,t_N)^\top\in\mathbb{R}^N
$$

denote the vector of observed targets, and let $X\in\mathbb{R}^{N\times D}$ be the design matrix whose rows are the feature vectors $x_n^\top$. Now, we assume our observations follow the model

$$
t_n=x_n^\top w+\epsilon_n,
\qquad
\epsilon_n\overset{\mathrm{i.i.d.}}{\sim}\mathcal{N}(0,\sigma^2).
$$

The likelihood of our data under Gaussian noise is

$$
p(t\mid X,w)
=\prod_{n=1}^{N}
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(t_n-x_n^\top w)^2}{2\sigma^2}\right).
$$

Place an independent Laplace prior on the coefficients $w$ using $\mu=0$ and $b=2\sigma^2/\lambda$ such that

$$
p(w)
=\prod_{d=1}^{D}
\frac{\lambda}{4\sigma^2}
\exp\left(-\frac{\lambda}{2\sigma^2}|w_d|\right).
$$

Recall that the **Maximum a Posteriori (MAP)** estimate chooses the parameter $w$ that maximizes the posterior density:

$$
\widehat{w}_{\mathrm{MAP}}
=\operatorname*{arg\,max}_{w}p(w\mid t,X)
=\operatorname*{arg\,max}_{w}p(t\mid X,w)p(w).
$$

Show that this MAP estimate is equivalent to the solution to the Lasso ($L_1$-regularized) regression problem.

**Answer (d):** Bayes' rule gives

$$
p(w\mid t,X)
=\frac{p(t\mid X,w)p(w)}{p(t\mid X)}.
$$

The evidence $p(t\mid X)$ does not depend on $w$. Maximizing the posterior is therefore equivalent to maximizing $p(t\mid X,w)p(w)$, or to minimizing its negative logarithm:

$$
\widehat{w}_{\mathrm{MAP}}
=\operatorname*{arg\,min}_{w}
\left[-\log p(t\mid X,w)-\log p(w)\right].
$$

For the Gaussian likelihood,

$$
\begin{aligned}
-\log p(t\mid X,w)
&=-\sum_{n=1}^{N}
\log\left[
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(t_n-x_n^\top w)^2}{2\sigma^2}\right)
\right]\\
&=\frac{N}{2}\log(2\pi\sigma^2)
+\frac{1}{2\sigma^2}
\sum_{n=1}^{N}(t_n-x_n^\top w)^2.
\end{aligned}
$$

Ignoring the first term, which is constant with respect to $w$,

$$
-\log p(t\mid X,w)
=\frac{1}{2\sigma^2}\lVert t-Xw\rVert_2^2
+\text{constant}.
$$

For the independent Laplace prior,

$$
\begin{aligned}
-\log p(w)
&=-\sum_{d=1}^{D}
\log\left[
\frac{\lambda}{4\sigma^2}
\exp\left(-\frac{\lambda}{2\sigma^2}|w_d|\right)
\right]\\
&=D\log\left(\frac{4\sigma^2}{\lambda}\right)
+\frac{\lambda}{2\sigma^2}
\sum_{d=1}^{D}|w_d|\\
&=\frac{\lambda}{2\sigma^2}\lVert w\rVert_1
+\text{constant}.
\end{aligned}
$$

Combining the two expressions gives

$$
\widehat{w}_{\mathrm{MAP}}
=\operatorname*{arg\,min}_{w}
\left[
\frac{1}{2\sigma^2}\lVert t-Xw\rVert_2^2
+\frac{\lambda}{2\sigma^2}\lVert w\rVert_1
\right].
$$

Multiplying the entire objective by the positive constant $\sigma^2$ does not change the minimizer, so

$$
\boxed{
\widehat{w}_{\mathrm{MAP}}
=\operatorname*{arg\,min}_{w}
\left[
\frac{1}{2}\lVert t-Xw\rVert_2^2
+\frac{\lambda}{2}\lVert w\rVert_1
\right]
}.
$$

This is the Lasso objective. Multiplying the entire objective by $2$ gives the equivalent common form

$$
\operatorname*{arg\,min}_{w}
\left[
\lVert t-Xw\rVert_2^2+\lambda\lVert w\rVert_1
\right].
$$

### (e)

Briefly contrast the two estimators:

1. **Laplace-noise MLE:** Minimizes the MAE ($L_1$ loss on residuals).

2. **Gaussian-noise + Laplace-prior MAP (Lasso):** Minimizes squared error with an $L_1$ penalty on coefficients.

**Answer (e):** The two estimators place the $L_1$ term on different quantities, so they have different effects.

1. For the **Laplace-noise MLE**, the objective is

   $$
   \sum_{n=1}^{N}|t_n-x_n^\top w|.
   $$

   The absolute-value penalty is applied to the residuals. Compared with squared error, it increases only linearly for a large residual, making the fit less sensitive to unusually large target errors. It does not directly encourage the coefficients in $w$ to be zero.

2. For the **Gaussian-noise + Laplace-prior MAP estimator**, the objective is

   $$
   \frac{1}{2}\lVert t-Xw\rVert_2^2
   +\frac{\lambda}{2}\lVert w\rVert_1.
   $$

   The data-fit term is squared error, while the absolute-value penalty is applied to the coefficients. This penalty shrinks the coefficients and can set some of them exactly to zero, producing a sparse model.

---

## 2. Bias-Variance Trade-off

As we saw in lecture, the expected error for a model $f_w$ at a test point $x$ can be decomposed as

$$
\begin{aligned}
\mathbb{E}\left[(t-f_w(x))^2\right]
&=\underbrace{\mathbb{E}\left[(t-h(x))^2\right]}_{\text{(i)}}
+\underbrace{\left(h(x)-\mathbb{E}[f_w(x)]\right)^2}_{\text{(ii)}}\\
&\quad+\underbrace{\mathbb{E}\left[\left(\mathbb{E}[f_w(x)]-f_w(x)\right)^2\right]}_{\text{(iii)}}.
\end{aligned}
\tag{1}
$$

Here, $f_w(x)$ is the prediction of our model, parameterized by $w$, trained on a particular dataset. We set $t$ to be the true label for the test point and assume it comes from some underlying ground-truth function $h$ such that

$$
t=h(x)+\epsilon,
$$

where $\epsilon$ is random noise inherent in the system. Assume

$$
\mathbb{E}[\epsilon]=0,
\qquad
\operatorname{Var}(\epsilon)=\sigma^2.
$$

### (a)

From the decomposition above, consider the following symbols:

- $t$: the true test label
- $w$: the weights of the fitted model
- $x$: the test input

Which of these are random variables, and which are not? Briefly explain.

**Answer (a):** In this decomposition, the expectations are taken over the random test noise and over the random choice of training dataset, while the test input $x$ is held fixed.

- The label $t$ is a random variable. Since

  $$
  t=h(x)+\epsilon,
  $$

  and $\epsilon$ is random, repeated observations at the same input $x$ may produce different values of $t$.

- The fitted weights $w$ are also treated as a random variable. The training dataset is randomly sampled, and fitting the model on a different dataset may produce different weights:

  $$
  \mathcal{D}_1\longmapsto w_1,
  \qquad
  \mathcal{D}_2\longmapsto w_2.
  $$

  Consequently, $f_w(x)$ is random even when $x$ is fixed. After training on one particular realized dataset, the fitted value of $w$ is fixed; its randomness here refers to variation across possible training datasets.

- The test input $x$ is not a random variable in this calculation. The decomposition measures prediction error at a particular fixed test point $x$.

Therefore,

$$
\boxed{
t\text{ and }w\text{ are random variables, while }x\text{ is fixed.}
}
$$

### (b)

Rewrite equation (1) in terms of $\sigma^2$, $b$ (the model's bias), and $v$ (the model's variance).

**Answer (b):** Consider each term in equation (1). In this question, $b$ denotes the model's bias; it is unrelated to the Laplace scale parameter $b$ used in Question 1.

For term (i), use $t=h(x)+\epsilon$:

$$
\begin{aligned}
\mathbb{E}\left[(t-h(x))^2\right]
&=\mathbb{E}[\epsilon^2]\\
&=\operatorname{Var}(\epsilon)
+\left(\mathbb{E}[\epsilon]\right)^2\\
&=\sigma^2.
\end{aligned}
$$

This is the irreducible noise. It remains even if the true function $h(x)$ is known exactly.

Define the model's bias at $x$ by

$$
b=\mathbb{E}[f_w(x)]-h(x).
$$

Term (ii) is therefore the squared bias:

$$
\left(h(x)-\mathbb{E}[f_w(x)]\right)^2
=(-b)^2=b^2.
$$

Define the model's variance at $x$ by

$$
v=\operatorname{Var}(f_w(x))
=\mathbb{E}\left[
\left(f_w(x)-\mathbb{E}[f_w(x)]\right)^2
\right].
$$

Because reversing the sign inside a square does not change its value, term (iii) is

$$
\mathbb{E}\left[
\left(\mathbb{E}[f_w(x)]-f_w(x)\right)^2
\right]
=v.
$$

Substituting the three terms into equation (1) gives

$$
\boxed{
\mathbb{E}\left[(t-f_w(x))^2\right]
=\sigma^2+b^2+v
}.
$$

Here, $\sigma^2$ is the irreducible noise, $b^2$ measures the model's systematic error, and $v$ measures how much its prediction changes across training datasets.

### (c)

Suppose you're tasked with predicting housing prices using a regression model.

#### (i)

If you move from linear regression to a neural network model with many more parameters, how do you expect the bias and variance to change?

**Answer (c)(i):** A neural network with many more parameters has greater flexibility than a linear model. It can represent more complex relationships between the housing features and prices, so its approximation error usually decreases:

$$
\boxed{\text{Bias decreases.}}
$$

The added flexibility also makes the fitted function more sensitive to the particular training sample. Two different training datasets may lead to noticeably different learned parameters and predictions:

$$
\boxed{\text{Variance increases.}}
$$

Thus, moving to the more flexible neural network generally trades lower bias for higher variance.

#### (ii)

What if you keep linear regression but add a ridge ($L_2$) regularization term? How does this affect bias and variance?

**Answer (c)(ii):** Ridge regression minimizes an objective of the form

$$
\lVert Xw-t\rVert_2^2+\lambda\lVert w\rVert_2^2.
$$

The regularization term discourages large coefficients and restricts the set of models that can be fitted. This restriction reduces the model's flexibility, so it may move the prediction farther from the true function:

$$
\boxed{\text{Bias increases.}}
$$

At the same time, shrinking the coefficients makes the fitted model less sensitive to changes in the training data:

$$
\boxed{\text{Variance decreases.}}
$$

As $\lambda$ increases, the usual effect is therefore higher bias and lower variance. If $\lambda$ becomes too large, the coefficients may be over-shrunk and the model may underfit.
