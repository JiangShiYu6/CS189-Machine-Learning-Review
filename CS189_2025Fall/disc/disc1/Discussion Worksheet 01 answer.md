# CS 189/289A - Discussion 1
## Fall 2025

---

> **Note:** Your TA will probably not cover all the problems on this worksheet. The discussion worksheets are not designed to be finished within an hour. They are deliberately made slightly longer so they can serve as resources you can use to practice, reinforce, and build upon concepts discussed in lectures, discussions, and homework.

## Chain Rule Reference

$$\frac{\partial}{\partial x}f(g(x) + h(x)) = \frac{\partial f}{\partial g}\frac{dg}{dx} + \frac{\partial f}{\partial h}\frac{dh}{dx} \quad (1)$$

---

## 1. Calculus

### (a) Sigmoid Function Derivatives

Consider the function
$$f(x, y) = \sigma(ax + by),$$
where $a, b \in \mathbb{R}$ and $\sigma(t) = \frac{1}{1 + e^{-t}}$ for $t \in \mathbb{R}$.

**(i)** Show that 
$$\frac{d\sigma}{dt} = \sigma(t)(1 - \sigma(t))$$

**Answer (i):**

Starting with the sigmoid function:
$$\sigma(t) = \frac{1}{1 + e^{-t}}$$

We compute the derivative using the chain rule. Let $u = 1 + e^{-t}$, so $\sigma(t) = u^{-1}$.

$$\frac{d\sigma}{dt} = \frac{d}{dt}(1 + e^{-t})^{-1}$$

Using the chain rule:
$$\frac{d\sigma}{dt} = -1 \cdot (1 + e^{-t})^{-2} \cdot (-e^{-t}) = \frac{e^{-t}}{(1 + e^{-t})^2}$$

Now we manipulate this expression:
$$\frac{d\sigma}{dt} = \frac{e^{-t}}{(1 + e^{-t})^2} = \frac{1}{1 + e^{-t}} \cdot \frac{e^{-t}}{1 + e^{-t}}$$

$$= \frac{1}{1 + e^{-t}} \cdot \left(\frac{1 + e^{-t} - 1}{1 + e^{-t}}\right)$$

$$= \frac{1}{1 + e^{-t}} \cdot \left(1 - \frac{1}{1 + e^{-t}}\right)$$

$$= \sigma(t) \cdot (1 - \sigma(t))$$

Therefore, $\boxed{\frac{d\sigma}{dt} = \sigma(t)(1 - \sigma(t))}$

---

**(ii)** Using the result you showed in part (i) and the chain rule, compute $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$.

**Answer (ii):**

Given: $f(x, y) = \sigma(ax + by)$

We apply the chain rule. Let $t = ax + by$, so $f = \sigma(t)$.

**Computing $\frac{\partial f}{\partial x}$:**

$$\frac{\partial f}{\partial x} = \frac{d\sigma}{dt} \cdot \frac{\partial t}{\partial x}$$

From part (i), we know $\frac{d\sigma}{dt} = \sigma(t)(1 - \sigma(t))$.

Also, $\frac{\partial t}{\partial x} = \frac{\partial}{\partial x}(ax + by) = a$.

Therefore:
$$\frac{\partial f}{\partial x} = \sigma(ax + by) \cdot (1 - \sigma(ax + by)) \cdot a$$

$$\boxed{\frac{\partial f}{\partial x} = a\sigma(ax + by)(1 - \sigma(ax + by))}$$

**Computing $\frac{\partial f}{\partial y}$:**

$$\frac{\partial f}{\partial y} = \frac{d\sigma}{dt} \cdot \frac{\partial t}{\partial y}$$

From part (i), $\frac{d\sigma}{dt} = \sigma(t)(1 - \sigma(t))$.

Also, $\frac{\partial t}{\partial y} = \frac{\partial}{\partial y}(ax + by) = b$.

Therefore:
$$\frac{\partial f}{\partial y} = \sigma(ax + by) \cdot (1 - \sigma(ax + by)) \cdot b$$

$$\boxed{\frac{\partial f}{\partial y} = b\sigma(ax + by)(1 - \sigma(ax + by))}$$

### (b) Partial Derivatives of Sum of Squares

For $x = \begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix} \in \mathbb{R}^n$, define
$$r(x) = \sum_{j=1}^{n} x_j^2$$

Compute the partial derivative $\frac{\partial r}{\partial x_i}$ for a generic coordinate $i \in \{1, \ldots, n\}$.

**Answer (b):**

To compute $\frac{\partial r}{\partial x_i}$, we take the partial derivative of $r(x)$ with respect to $x_i$, treating all other variables as constants:

$$\frac{\partial r}{\partial x_i} = \frac{\partial}{\partial x_i}\left(\sum_{j=1}^{n} x_j^2\right)$$

$$= \sum_{j=1}^{n} \frac{\partial}{\partial x_i}(x_j^2)$$

Since $\frac{\partial}{\partial x_i}(x_j^2) = 0$ for all $j \neq i$ (as $x_j$ is independent of $x_i$), and $\frac{\partial}{\partial x_i}(x_i^2) = 2x_i$, we have:

$$\frac{\partial r}{\partial x_i} = 2x_i$$

Therefore, $\boxed{\frac{\partial r}{\partial x_i} = 2x_i}$ for $i \in \{1, \ldots, n\}$.

### (c) Partial Derivatives of Linear Function

Let $w \in \mathbb{R}^n$ be a constant vector and define the scalar function
$$s(x) = w^\top x = \sum_{j=1}^{n} w_j x_j$$

Compute $\frac{\partial s}{\partial x_i}$ for a generic coordinate $i \in \{1, \ldots, n\}$.

**Answer (c):**

To compute $\frac{\partial s}{\partial x_i}$, we take the partial derivative of $s(x)$ with respect to $x_i$, treating all other variables as constants:

$$\frac{\partial s}{\partial x_i} = \frac{\partial}{\partial x_i}\left(\sum_{j=1}^{n} w_j x_j\right)$$

$$= \sum_{j=1}^{n} w_j \frac{\partial x_j}{\partial x_i}$$

Since $\frac{\partial x_j}{\partial x_i} = \delta_{ij}$ (Kronecker delta: 1 if $i = j$, 0 otherwise), we have:

$$\frac{\partial s}{\partial x_i} = \sum_{j=1}^{n} w_j \delta_{ij} = w_i$$

Therefore, $\boxed{\frac{\partial s}{\partial x_i} = w_i}$ for $i \in \{1, \ldots, n\}$.

**Alternative approach:** The gradient of $s(x) = w^\top x$ is $\nabla s = w$, so the partial derivative with respect to the $i$-th coordinate is simply the $i$-th component of $w$.

---

## 2. Linear Algebra

### (a) Symmetry of $A^\top A$

Prove that $A^\top A$ is symmetric for any $A \in \mathbb{R}^{m \times n}$.

**Answer (a):**

To prove that $A^\top A$ is symmetric, we need to show that $(A^\top A)^\top = A^\top A$.

Using the property that $(XY)^\top = Y^\top X^\top$:

$$(A^\top A)^\top = A^\top (A^\top)^\top = A^\top A$$

Since $(A^\top)^\top = A$ by definition of matrix transpose.

Therefore, $\boxed{A^\top A \text{ is symmetric}}$ ✓

---

### (b) Singular Values

Consider the matrix
$$B = \begin{pmatrix} 1 & 0 \\ 0 & 2 \\ 2 & 1 \end{pmatrix}$$

Find the singular values of $B$.

**Answer (b):**

The singular values of $B$ are the square roots of the eigenvalues of $B^\top B$.

**Step 1: Compute $B^\top B$**

$$B^\top = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 2 & 1 \end{pmatrix}$$

$$B^\top B = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 2 & 1 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 2 \\ 2 & 1 \end{pmatrix}$$

Computing the product:
- $(B^\top B)_{11} = 1 \cdot 1 + 0 \cdot 0 + 2 \cdot 2 = 1 + 4 = 5$
- $(B^\top B)_{12} = 1 \cdot 0 + 0 \cdot 2 + 2 \cdot 1 = 2$
- $(B^\top B)_{21} = 0 \cdot 1 + 2 \cdot 0 + 1 \cdot 2 = 2$
- $(B^\top B)_{22} = 0 \cdot 0 + 2 \cdot 2 + 1 \cdot 1 = 4 + 1 = 5$

$$B^\top B = \begin{pmatrix} 5 & 2 \\ 2 & 5 \end{pmatrix}$$

**Step 2: Find the eigenvalues of $B^\top B$**

The characteristic polynomial is:
$$\det(B^\top B - \lambda I) = \det\begin{pmatrix} 5 - \lambda & 2 \\ 2 & 5 - \lambda \end{pmatrix}$$

$$= (5 - \lambda)^2 - 4 = 25 - 10\lambda + \lambda^2 - 4$$

$$= \lambda^2 - 10\lambda + 21 = 0$$

Factoring:
$$(\lambda - 3)(\lambda - 7) = 0$$

So $\lambda_1 = 3$ and $\lambda_2 = 7$.

**Step 3: Compute the singular values**

The singular values are:
$$\sigma_1 = \sqrt{\lambda_1} = \sqrt{3}$$
$$\sigma_2 = \sqrt{\lambda_2} = \sqrt{7}$$

Therefore, the singular values of $B$ are $\boxed{\sigma_1 = \sqrt{3}, \quad \sigma_2 = \sqrt{7}}$

---

## 3. Probability Review

An incoming email is spam with prior $p(S) = 0.2$ and not spam with $p(\bar{S}) = 0.8$. Two independent filters flag spam:

$$p(F_1 = 1 | S) = 0.9, \quad p(F_1 = 1 | \bar{S}) = 0.1,$$
$$p(F_2 = 1 | S) = 0.8, \quad p(F_2 = 1 | \bar{S}) = 0.05,$$

and $F_1, F_2$ are independent given the class ($S$ or $\bar{S}$).

### (a) Joint Probability of Both Filters Flagging

What is the probability that both filters flag an email as spam?

**Answer (a):**

We need to find $p(F_1 = 1, F_2 = 1)$ — the probability that both filters flag an email.

Using the law of total probability, we condition on whether the email is spam:

$$p(F_1 = 1, F_2 = 1) = p(F_1 = 1, F_2 = 1 | S) \cdot p(S) + p(F_1 = 1, F_2 = 1 | \bar{S}) \cdot p(\bar{S})$$

Since $F_1$ and $F_2$ are independent given the class:

$$p(F_1 = 1, F_2 = 1 | S) = p(F_1 = 1 | S) \cdot p(F_2 = 1 | S) = 0.9 \times 0.8 = 0.72$$

$$p(F_1 = 1, F_2 = 1 | \bar{S}) = p(F_1 = 1 | \bar{S}) \cdot p(F_2 = 1 | \bar{S}) = 0.1 \times 0.05 = 0.005$$

Therefore:

$$p(F_1 = 1, F_2 = 1) = 0.72 \times 0.2 + 0.005 \times 0.8$$

$$= 0.144 + 0.004 = 0.148$$

$$\boxed{p(F_1 = 1, F_2 = 1) = 0.148 \text{ or } \frac{148}{1000} = \frac{37}{250}}$$

---

### (b) Posterior Probability (Bayes' Theorem)

Given that both filters flag an email, what is the probability of the email being spam? (You can leave your answer as an unsimplified fraction.)

**Answer (b):**

We need to find $p(S | F_1 = 1, F_2 = 1)$ — the posterior probability that an email is spam given that both filters flagged it.

Using Bayes' theorem:

$$p(S | F_1 = 1, F_2 = 1) = \frac{p(F_1 = 1, F_2 = 1 | S) \cdot p(S)}{p(F_1 = 1, F_2 = 1)}$$

From part (a), we have:
- $p(F_1 = 1, F_2 = 1 | S) = 0.72$
- $p(S) = 0.2$
- $p(F_1 = 1, F_2 = 1) = 0.148$

Computing the numerator:
$$p(F_1 = 1, F_2 = 1 | S) \cdot p(S) = 0.72 \times 0.2 = 0.144 = \frac{144}{1000}$$

Therefore:
$$p(S | F_1 = 1, F_2 = 1) = \frac{0.144}{0.148} = \frac{144}{148}$$

Simplifying by dividing both numerator and denominator by their GCD (which is 4):
$$\frac{144}{148} = \frac{36}{37}$$

$$\boxed{p(S | F_1 = 1, F_2 = 1) = \frac{144}{148} \text{ or } \frac{36}{37}}$$

**Interpretation:** Given that both filters flagged an email as spam, there is approximately a 97.3% probability ($36/37 \approx 0.973$) that the email is actually spam. The dual filter flagging significantly increases our confidence that the email is spam compared to the prior probability of 20%.
