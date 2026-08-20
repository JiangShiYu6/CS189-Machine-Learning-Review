# CS 189/289A - Discussion 3
## Fall 2025

---

> **Note:** Your TA will probably not cover all the problems on this worksheet. The discussion worksheets are not designed to be finished within an hour. They are deliberately made slightly longer so they can serve as resources you can use to practice, reinforce, and build upon concepts discussed in lectures, discussions, and homework.

## 1. MLE with a Linear Constraint on Independent Gaussians

### (a) Conditional MLE

Let

$$
X \sim \mathcal{N}(\mu_X, \sigma_X^2),
\qquad
Y \sim \mathcal{N}(\mu_Y, \sigma_Y^2)
$$

be independent. We observe only their sum

$$
S := X + Y
$$

and are told that $S=2$.

#### (i) Show that $p_{X\mid S}(x\mid 2) \propto p_X(x)p_Y(2-x)$

**Answer (i):**

By the definition of conditional density,

$$
p_{X\mid S}(x\mid 2)
= \frac{p_{X,S}(x,2)}{p_S(2)}.
$$

Because $S=X+Y$, once $X=x$ and $S=2$ are fixed, we must have

$$
Y=2-x.
$$

Under the change of variables $(X,Y)\mapsto(X,S)$, where $S=X+Y$, the absolute value of the Jacobian determinant is $1$. Therefore,

$$
p_{X,S}(x,2)
=p_{X,Y}(x,2-x).
$$

Since $X$ and $Y$ are independent,

$$
p_{X,Y}(x,2-x)=p_X(x)p_Y(2-x).
$$

It follows that

$$
p_{X\mid S}(x\mid 2)
=\frac{p_X(x)p_Y(2-x)}{p_S(2)}.
$$

For the fixed observation $S=2$, the denominator $p_S(2)$ does not depend on $x$, so it is a proportionality constant with respect to $x$. Hence,

$$
\boxed{p_{X\mid S}(x\mid 2) \propto p_X(x)p_Y(2-x)}.
$$

---

#### (ii) Convert likelihood maximization into quadratic minimization

**Answer (ii):**

The two Gaussian densities are

$$
p_X(x)
= \frac{1}{\sqrt{2\pi\sigma_X^2}}
\exp\left(-\frac{(x-\mu_X)^2}{2\sigma_X^2}\right)
$$

and

$$
p_Y(2-x)
= \frac{1}{\sqrt{2\pi\sigma_Y^2}}
\exp\left(-\frac{(2-x-\mu_Y)^2}{2\sigma_Y^2}\right).
$$

Multiplying them gives

$$
p_{X\mid S}(x\mid 2)
\propto
\exp\left(
-\frac{(x-\mu_X)^2}{2\sigma_X^2}
-\frac{(2-x-\mu_Y)^2}{2\sigma_Y^2}
\right).
$$

The exponential function is strictly increasing, so maximizing this expression is equivalent to maximizing its exponent. Because the exponent is the negative of a sum, this is equivalent to minimizing

$$
\boxed{
\frac{(x-\mu_X)^2}{2\sigma_X^2}
+
\frac{(2-x-\mu_Y)^2}{2\sigma_Y^2}
}.
$$

---

#### (iii) Find $\widehat{x}_{\mathrm{MLE}}$

**Answer (iii):**

Define

$$
f(x)
= \frac{(x-\mu_X)^2}{2\sigma_X^2}
+ \frac{(2-x-\mu_Y)^2}{2\sigma_Y^2}.
$$

Differentiate with respect to $x$:

$$
f'(x)
= \frac{x-\mu_X}{\sigma_X^2}
- \frac{2-x-\mu_Y}{\sigma_Y^2}.
$$

Setting the derivative equal to zero gives

$$
\frac{x-\mu_X}{\sigma_X^2}
= \frac{2-x-\mu_Y}{\sigma_Y^2}.
$$

After multiplying through by $\sigma_X^2\sigma_Y^2$ and collecting the terms containing $x$,

$$
\sigma_Y^2(x-\mu_X)
= \sigma_X^2(2-x-\mu_Y),
$$

$$
(\sigma_X^2+\sigma_Y^2)x
= \sigma_Y^2\mu_X+\sigma_X^2(2-\mu_Y).
$$

Thus,

$$
\boxed{
\widehat{x}_{\mathrm{MLE}}
= \frac{\sigma_Y^2\mu_X+\sigma_X^2(2-\mu_Y)}
{\sigma_X^2+\sigma_Y^2}
}.
$$

Equivalently, in precision-weighted form,

$$
\boxed{
\widehat{x}_{\mathrm{MLE}}
=
\frac{\dfrac{\mu_X}{\sigma_X^2}
+\dfrac{2-\mu_Y}{\sigma_Y^2}}
{\dfrac{1}{\sigma_X^2}+\dfrac{1}{\sigma_Y^2}}
}.
$$

This is a weighted average of the two estimates $\mu_X$ and $2-\mu_Y$. The estimate associated with the smaller variance receives the larger weight.

The second derivative is

$$
f''(x)=\frac{1}{\sigma_X^2}+\frac{1}{\sigma_Y^2}>0,
$$

so this stationary point is the unique minimizer of $f$ and therefore the unique maximizer of the likelihood.

---

#### (iv) Special case: $\mu_X=\mu_Y=0$ and $\sigma_X^2=\sigma_Y^2$

**Answer (iv):**

Let $\sigma_X^2=\sigma_Y^2=\sigma^2$. Substituting into the expression from part (iii),

$$
\widehat{x}_{\mathrm{MLE}}
= \frac{\sigma^2(0)+\sigma^2(2-0)}{\sigma^2+\sigma^2}
= \frac{2\sigma^2}{2\sigma^2}
=1.
$$

Therefore,

$$
\boxed{\widehat{x}_{\mathrm{MLE}}=1}.
$$

By symmetry, the most likely explanation of $X+Y=2$ is that the two identically distributed variables each contribute half of the observed sum.

---

## 2. Binomial MLE with Misclassification

Suppose each true Bernoulli trial has success probability $p$. A true success is recorded as a success with probability $1-q$, while a true failure is incorrectly recorded as a success with probability $r$. There are $m$ independent experiments, each containing $n$ independent trials, and $Y_i$ is the number of recorded successes in experiment $i$.

### (a) Derive the likelihood for $D=\{y_i\}_{i=1}^m$

**Answer (a):**

Let $Z$ denote the true result of one trial and let $W$ denote its recorded result. By the law of total probability, the probability that the device records a success is

$$
\begin{aligned}
\theta(p)
&:= \Pr(W=1) \\
&= \Pr(W=1\mid Z=1)\Pr(Z=1)
+\Pr(W=1\mid Z=0)\Pr(Z=0) \\
&= (1-q)p+r(1-p) \\
&= r+(1-q-r)p.
\end{aligned}
$$

After marginalizing over the unobserved true outcomes, every recorded trial is therefore Bernoulli with success probability $\theta(p)$. Because the trials are independent,

$$
Y_i\mid p \sim \operatorname{Bin}\bigl(n,\theta(p)\bigr).
$$

Thus,

$$
\Pr(Y_i=y_i\mid p)
= \binom{n}{y_i}
\theta(p)^{y_i}
\bigl(1-\theta(p)\bigr)^{n-y_i}.
$$

The $m$ experiments are independent, so the likelihood is

$$
\boxed{
L(p;D)
= \prod_{i=1}^{m}
\binom{n}{y_i}
\left[r+(1-q-r)p\right]^{y_i}
\left[1-r-(1-q-r)p\right]^{n-y_i}
},
$$

for $0\le p\le1$.

If

$$
T:=\sum_{i=1}^{m}y_i
\qquad\text{and}\qquad
N:=mn,
$$

then, after collecting all factors that do not depend on $p$,

$$
L(p;D)
\propto
\theta(p)^T\bigl(1-\theta(p)\bigr)^{N-T}.
$$

---

### (b) Find the MLE $\widehat{p}$ in closed form

**Answer (b):**

Ignoring additive constants that do not depend on $p$, the log-likelihood is

$$
\ell(p)
= T\log\theta(p)+(N-T)\log\bigl(1-\theta(p)\bigr),
$$

where

$$
\theta(p)=r+(1-q-r)p.
$$

Let

$$
a:=1-q-r.
$$

Then $\theta'(p)=a$, and

$$
\begin{aligned}
\ell'(p)
&=a\left(\frac{T}{\theta(p)}-
\frac{N-T}{1-\theta(p)}\right) \\
&=a\frac{T-N\theta(p)}
{\theta(p)(1-\theta(p))}.
\end{aligned}
$$

Assume first that $a\ne0$ and the maximizer is in the interior. Setting $\ell'(p)=0$ gives

$$
T-N\theta(p)=0,
$$

so

$$
\widehat{\theta}=\frac{T}{N}
=\frac{1}{mn}\sum_{i=1}^{m}y_i.
$$

Solving $\widehat{\theta}=r+ap$ for $p$ gives the unconstrained estimate

$$
\widetilde{p}
=\frac{\dfrac{1}{mn}\sum_{i=1}^{m}y_i-r}
{1-q-r}.
$$

Moreover,

$$
\ell''(p)
=-a^2\left(
\frac{T}{\theta(p)^2}
+\frac{N-T}{(1-\theta(p))^2}
\right)\le0,
$$

so the log-likelihood is concave in $p$. Enforcing the parameter constraint $0\le p\le1$ therefore gives

$$
\boxed{
\widehat{p}
=\min\left\{1,
\max\left\{0,
\frac{\dfrac{1}{mn}\sum_{i=1}^{m}y_i-r}
{1-q-r}
\right\}
\right\}
},
\qquad q+r\ne1.
$$

When the unconstrained estimate already lies in $[0,1]$, this simplifies to

$$
\boxed{
\widehat{p}
=\frac{\dfrac{1}{mn}\sum_{i=1}^{m}y_i-r}
{1-q-r}
}.
$$

Finally, if $q+r=1$, then

$$
\theta(p)=r
$$

for every $p$. In that case the distribution of the observations does not depend on $p$, so $p$ is not identifiable and every $p\in[0,1]$ is an MLE.

---

## 3. Proof of K-means Convergence

The K-means objective is

$$
J(\{r_{nk}\},\{\mu_k\})
=\sum_{n=1}^{N}\sum_{k=1}^{K}
r_{nk}\lVert x_n-\mu_k\rVert^2.
$$

**Answer:**

We show that neither step of K-means increases $J$, and then use the fact that there are only finitely many possible cluster assignments.

### (A) The objective never increases

**Assignment step.** Fix the current centers $\{\mu_k\}_{k=1}^K$. For each point $x_n$, K-means assigns it to a nearest center, so its contribution to the objective becomes

$$
\min_{1\le k\le K}\lVert x_n-\mu_k\rVert^2.
$$

This value is no larger than the contribution under the previous assignment. Therefore,

$$
J(R^{(t+1)},M^{(t)})
\le J(R^{(t)},M^{(t)}),
$$

where $R$ denotes the assignments and $M$ denotes the centers.

**Update step.** Now fix the assignments. For a nonempty cluster $k$, define

$$
C_k=\{n:r_{nk}=1\},
\qquad N_k=|C_k|.
$$

Its contribution to the objective is

$$
J_k(\mu)=\sum_{n\in C_k}\lVert x_n-\mu\rVert^2.
$$

Differentiating with respect to $\mu$ gives

$$
\nabla_\mu J_k(\mu)
=2N_k\mu-2\sum_{n\in C_k}x_n.
$$

Setting this gradient equal to zero yields the unique minimizer

$$
\mu_k^*
=\frac{1}{N_k}\sum_{n\in C_k}x_n
=\frac{\sum_{n=1}^{N}r_{nk}x_n}
{\sum_{n=1}^{N}r_{nk}}.
$$

Thus, replacing each center by the mean of its assigned points cannot increase $J$:

$$
J(R^{(t+1)},M^{(t+1)})
\le J(R^{(t+1)},M^{(t)}).
$$

Combining the two inequalities gives

$$
\boxed{
J(R^{(t+1)},M^{(t+1)})
\le J(R^{(t)},M^{(t)})
}.
$$

Hence the objective is monotonically nonincreasing.

### (B) Only finitely many assignments exist

Each of the $N$ data points can be assigned to one of $K$ clusters. Therefore, there are at most

$$
K^N
$$

possible assignment matrices $R$. Once an assignment is fixed, the update step uniquely determines the mean of every nonempty cluster. Thus, after each complete iteration, the algorithm can be in only finitely many assignment-center states.

After a complete iteration, either $J$ decreases strictly or it remains unchanged. If it remains unchanged, equality in the update-step inequality means that every nonempty center was already the mean of its newly assigned points, so the centers do not move. Applying the deterministic assignment rule again to these same centers produces the same assignments. The algorithm has therefore reached a fixed point.

Every nonterminal iteration strictly decreases the objective among a finite set of possible post-update states. Such decreases can occur only finitely many times, so K-means must eventually reach a state where neither step changes anything. Therefore,

$$
\boxed{\text{K-means converges after a finite number of iterations.}}
$$

This argument assumes a fixed rule for breaking equal-distance ties and handling empty clusters. These rules prevent the implementation from cycling between equivalent states with the same objective value.
