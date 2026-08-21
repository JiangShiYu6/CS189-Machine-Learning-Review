# CS 189/289A - Discussion 6

## Fall 2025

---

> **Note:** Your TA will probably not cover all the problems on this worksheet. The discussion worksheets are not designed to be finished within an hour. They are deliberately made slightly longer so they can serve as resources you can use to practice, reinforce, and build upon concepts discussed in lectures, discussions, and homework.

## This week's cool AI demo/video

[Robot dog](https://www.youtube.com/watch?v=iI8UUu9g8iI)

[Super fast recovery on humanoids](https://www.youtube.com/watch?v=bPSLMX_V38E)

[Ping pong with humanoids](https://www.youtube.com/watch?v=tOfPKW6D3gE)

---

## 1. Evaluation Metrics and Threshold Selection

You trained a logistic regression model that outputs a probability $\widehat{p}(y=1\mid x)$ for each test example. The following table shows the true labels and predicted probabilities for a small test set:

| ID | True $y$ | $\widehat{p}$ |
|---:|---:|---:|
| 1 | 1 | 0.93 |
| 2 | 1 | 0.84 |
| 3 | 0 | 0.72 |
| 4 | 0 | 0.63 |
| 5 | 0 | 0.58 |
| 6 | 1 | 0.49 |
| 7 | 0 | 0.41 |
| 8 | 0 | 0.35 |
| 9 | 1 | 0.32 |
| 10 | 0 | 0.18 |

There are 4 positive and 6 negative examples. The model predicts $\widehat{y}=1$ when $\widehat{p}\ge\tau$, where $\tau\in(0,1)$ is the binary classification threshold. Recall that for binary classification, a prediction can be a true positive (TP), true negative (TN), false positive (FP), or false negative (FN).

For this problem, we will consider three metrics to evaluate classification performance:

$$
\operatorname{Accuracy}
=\frac{TP+TN}{TP+FP+TN+FN},
$$

$$
\operatorname{Precision}
=\frac{TP}{TP+FP},
\qquad
\operatorname{Recall}
=\frac{TP}{TP+FN}.
$$

### (a)

Explain in words the difference between accuracy, precision, and recall for binary classification.

**Answer (a):** Accuracy measures the fraction of all predictions that are correct. It counts both correctly identified positive examples and correctly identified negative examples:

$$
\operatorname{Accuracy}
=\frac{TP+TN}{TP+FP+TN+FN}.
$$

Precision looks only at the examples predicted as positive. It asks: among all examples for which the model predicted $\widehat{y}=1$, what fraction are actually positive?

$$
\operatorname{Precision}
=\frac{TP}{TP+FP}.
$$

A high precision means that positive predictions are usually correct, so the model produces few false positives.

Recall looks only at the examples that are actually positive. It asks: among all examples with $y=1$, what fraction did the model identify?

$$
\operatorname{Recall}
=\frac{TP}{TP+FN}.
$$

A high recall means that the model misses few positive examples, so it produces few false negatives. Precision conditions on the predicted positive class, while recall conditions on the actual positive class.

### (b)

For each threshold $\tau\in\{0.3,0.5,0.8\}$, compute the elements of the binary confusion matrix: TP, FP, TN, and FN. Then use these values to calculate accuracy, precision, and recall.

**Answer (b):** The prediction rule is $\widehat{y}=1$ when $\widehat{p}\ge\tau$.

For $\tau=0.3$, IDs 1 through 9 are predicted positive. The classifications are

$$
\begin{aligned}
TP&=4 &&\text{(IDs 1, 2, 6, 9)},\\
FP&=5 &&\text{(IDs 3, 4, 5, 7, 8)},\\
TN&=1 &&\text{(ID 10)},\\
FN&=0.
\end{aligned}
$$

Therefore,

$$
\begin{aligned}
\operatorname{Accuracy}
&=\frac{4+1}{10}=0.50,\\
\operatorname{Precision}
&=\frac{4}{4+5}=\frac{4}{9}\approx0.44,\\
\operatorname{Recall}
&=\frac{4}{4+0}=1.00.
\end{aligned}
$$

For $\tau=0.5$, IDs 1 through 5 are predicted positive. The classifications are

$$
\begin{aligned}
TP&=2 &&\text{(IDs 1, 2)},\\
FP&=3 &&\text{(IDs 3, 4, 5)},\\
TN&=3 &&\text{(IDs 7, 8, 10)},\\
FN&=2 &&\text{(IDs 6, 9)}.
\end{aligned}
$$

Therefore,

$$
\begin{aligned}
\operatorname{Accuracy}
&=\frac{2+3}{10}=0.50,\\
\operatorname{Precision}
&=\frac{2}{2+3}=0.40,\\
\operatorname{Recall}
&=\frac{2}{2+2}=0.50.
\end{aligned}
$$

For $\tau=0.8$, only IDs 1 and 2 are predicted positive. The classifications are

$$
\begin{aligned}
TP&=2 &&\text{(IDs 1, 2)},\\
FP&=0,\\
TN&=6 &&\text{(IDs 3, 4, 5, 7, 8, 10)},\\
FN&=2 &&\text{(IDs 6, 9)}.
\end{aligned}
$$

Therefore,

$$
\begin{aligned}
\operatorname{Accuracy}
&=\frac{2+6}{10}=0.80,\\
\operatorname{Precision}
&=\frac{2}{2+0}=1.00,\\
\operatorname{Recall}
&=\frac{2}{2+2}=0.50.
\end{aligned}
$$

The results can be summarized as follows:

| $\tau$ | TP | FP | TN | FN | Accuracy | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 4 | 5 | 1 | 0 | 0.50 | $4/9\approx0.44$ | 1.00 |
| 0.5 | 2 | 3 | 3 | 2 | 0.50 | 0.40 | 0.50 |
| 0.8 | 2 | 0 | 6 | 2 | 0.80 | 1.00 | 0.50 |

### (c)

How would you choose the best threshold for this dataset?

#### (i)

Consider a disease screening scenario, where $y=1$ indicates that a patient actually has the disease and $\widehat{y}=1$ indicates that we detect the disease and provide treatment. Which threshold from part (b) would you choose? Provide a brief justification for your answer.

**Answer (c)(i):** In disease screening, a false negative means failing to detect a patient who actually has the disease. This error is usually more costly than a false positive, since a false positive can be checked with further testing. The threshold should therefore prioritize recall.

From part (b),

$$
\begin{array}{c|ccc}
\tau & 0.3 & 0.5 & 0.8\\ \hline
\operatorname{Recall} & 1.00 & 0.50 & 0.50
\end{array}
$$

Thus, the best choice among the three candidates is

$$
\boxed{\tau=0.3}.
$$

At this threshold, $TP=4$ and $FN=0$, so every patient with the disease is detected. The tradeoff is a larger number of false positives.

#### (ii)

How does your answer change if the context is spam detection, where $y=1$ indicates an email is spam and $\widehat{y}=1$ indicates that we detect spam and immediately delete it?

**Answer (c)(ii):** In this setting, a false positive means deleting a legitimate email. Since deletion happens immediately, false positives are more costly than false negatives, which merely allow some spam into the inbox. The threshold should therefore prioritize precision.

From part (b),

$$
\begin{array}{c|ccc}
\tau & 0.3 & 0.5 & 0.8\\ \hline
\operatorname{Precision} & 0.44 & 0.40 & 1.00
\end{array}
$$

Thus, the best choice among the three candidates is

$$
\boxed{\tau=0.8}.
$$

At this threshold, $FP=0$, so no legitimate email in this dataset is deleted. The tradeoff is lower recall: two spam emails are missed.

---

## 2. Statistical Justification of Logistic Regression

Assume that we have $N$ i.i.d. data points

$$
\mathcal{D}=\{(x_n,y_n)\}_{n=1}^{N},
$$

where each $y_n$ is a binary label in $\{0,1\}$. We model the posterior probability of the labels given the observed features as a Bernoulli distribution, where the probability of a positive sample is given by the sigmoid function:

$$
p(Y=y\mid x;w)=p^y(1-p)^{1-y},
$$

where

$$
p=\sigma(w^\top x),
\qquad
\sigma(a)=\frac{1}{1+e^{-a}}.
$$

### (a)

Show that for a given data point $x$, the log ratio of the conditional probabilities, or log odds, is linear in $x$. More specifically, show that

$$
\log\frac{p(Y=1\mid x;w)}{p(Y=0\mid x;w)}
=w^\top x.
$$

**Answer (a):** Let

$$
z=w^\top x
\qquad\text{and}\qquad
p=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

For a Bernoulli label, the two conditional class probabilities are

$$
p(Y=1\mid x;w)=p
=\frac{1}{1+e^{-z}}
$$

and

$$
\begin{aligned}
p(Y=0\mid x;w)
&=1-p\\
&=1-\frac{1}{1+e^{-z}}\\
&=\frac{e^{-z}}{1+e^{-z}}.
\end{aligned}
$$

Their ratio, called the odds of class $1$ against class $0$, is

$$
\begin{aligned}
\frac{p(Y=1\mid x;w)}{p(Y=0\mid x;w)}
&=\frac{\dfrac{1}{1+e^{-z}}}
{\dfrac{e^{-z}}{1+e^{-z}}}\\
&=\frac{1}{e^{-z}}\\
&=e^z.
\end{aligned}
$$

Taking the natural logarithm gives

$$
\begin{aligned}
\log\frac{p(Y=1\mid x;w)}{p(Y=0\mid x;w)}
&=\log(e^z)\\
&=z\\
&=\boxed{w^\top x}.
\end{aligned}
$$

The final result says that logistic regression is linear in the log odds, even though the probability $p=\sigma(w^\top x)$ is a nonlinear function of $x$. Expanding the inner product,

$$
w^\top x=\sum_{j=1}^{D}w_jx_j.
$$

If feature $x_j$ increases by one unit while all other features remain fixed, the log odds change by $w_j$. Equivalently, the odds are multiplied by $e^{w_j}$:

$$
\frac{\operatorname{odds}(x_j+1)}
{\operatorname{odds}(x_j)}=e^{w_j}.
$$

The sign of $w^\top x$ also has a direct interpretation:

$$
\begin{array}{c|c|c}
w^\top x & \text{Odds} & \text{More likely class}\\ \hline
>0 & >1 & Y=1\\
=0 & =1 & \text{Neither; }p=0.5\\
<0 & <1 & Y=0
\end{array}
$$

For the usual threshold $p=0.5$, the decision boundary is therefore

$$
\boxed{w^\top x=0},
$$

which is a linear hyperplane in the feature space.

### (b)

Starting from the Bernoulli likelihood, derive the logistic loss using Maximum Likelihood Estimation (MLE). Show that maximizing the likelihood of a Bernoulli model is equivalent to minimizing the logistic loss.

**Answer (b):** For observation $n$, define

$$
z_n=w^\top x_n,
\qquad
p_n=\sigma(z_n)=p(Y=1\mid x_n;w).
$$

Because $y_n\in\{0,1\}$, its Bernoulli conditional probability can represent both possible labels in one expression:

$$
p(Y=y_n\mid x_n;w)
=p_n^{y_n}(1-p_n)^{1-y_n}.
$$

Indeed, if $y_n=1$, this expression equals $p_n$; if $y_n=0$, it equals $1-p_n$.

The observations are i.i.d., so their joint likelihood is

$$
\begin{aligned}
p(\mathcal{D}\mid w)
&=\prod_{n=1}^{N}p(Y=y_n\mid x_n;w)\\
&=\prod_{n=1}^{N}
p_n^{y_n}(1-p_n)^{1-y_n}.
\end{aligned}
$$

Taking the natural logarithm converts the product into a sum:

$$
\begin{aligned}
\ell(w)
&=\log p(\mathcal{D}\mid w)\\
&=\sum_{n=1}^{N}
\log\left[p_n^{y_n}(1-p_n)^{1-y_n}\right]\\
&=\sum_{n=1}^{N}
\left[y_n\log p_n+(1-y_n)\log(1-p_n)\right].
\end{aligned}
$$

The logarithm is strictly increasing, so maximizing the likelihood and maximizing the log-likelihood produce the same parameter estimate. Maximizing a function is also equivalent to minimizing its negative. Therefore,

$$
\begin{aligned}
\widehat{w}_{\mathrm{MLE}}
&=\operatorname*{arg\,max}_{w}p(\mathcal{D}\mid w)\\
&=\operatorname*{arg\,max}_{w}\ell(w)\\
&=\operatorname*{arg\,min}_{w}\bigl[-\ell(w)\bigr].
\end{aligned}
$$

The negative log-likelihood is

$$
\boxed{
\mathcal{L}_{\mathrm{logistic}}(w)
=-\sum_{n=1}^{N}
\left[
y_n\log p_n+(1-y_n)\log(1-p_n)
\right]
}.
$$

This is the logistic loss, also called binary cross-entropy. It can be written directly in terms of $w^\top x_n$. Since

$$
\log p_n
=z_n-\log(1+e^{z_n})
$$

and

$$
\log(1-p_n)
=-\log(1+e^{z_n}),
$$

the loss for one observation becomes

$$
\begin{aligned}
-\left[y_n\log p_n+(1-y_n)\log(1-p_n)\right]
&=-y_n\left[z_n-\log(1+e^{z_n})\right]\\
&\quad+(1-y_n)\log(1+e^{z_n})\\
&=\log(1+e^{z_n})-y_nz_n.
\end{aligned}
$$

Hence an equivalent form of the full objective is

$$
\boxed{
\mathcal{L}_{\mathrm{logistic}}(w)
=\sum_{n=1}^{N}
\left[
\log\left(1+e^{w^\top x_n}\right)
-y_nw^\top x_n
\right]
}.
$$

For a positive example ($y_n=1$), the loss is $-\log p_n$; for a negative example ($y_n=0$), it is $-\log(1-p_n)$. A confident incorrect prediction assigns a very small probability to the observed label and therefore receives a large loss.
