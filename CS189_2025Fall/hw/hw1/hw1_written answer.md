# CS 189/289A - Homework 1
## Fall 2025

**Due:** Friday, September 19 at 11:59 pm

---

**Deliverables.** Submit a single PDF of your write-up to Gradescope *HW1 Write-Up*.

**Start each problem on a new page.**

---

## Honor Code

*Write and sign the following statement:*

> "I certify that all solutions in this document are entirely my own and that I have not looked at anyone else's solution. I have given credit to all external sources I consulted."

---

## Linear Algebra

### 1. System of Equations (3 points)

Consider the linear system

$$
\begin{bmatrix}
1 & 2 & 3\\
2 & 0 & 6\\
3 & 6 & 9
\end{bmatrix}
\begin{bmatrix}
x\\
y\\
z
\end{bmatrix}
=
\begin{bmatrix}
6\\
8\\
12
\end{bmatrix}.
$$

How many solutions does this system have? Explain your reasoning.

**Answer:**

There are **no solutions**.

The third equation has left-hand side equal to three times the first equation's left-hand side:

$$
3x + 6y + 9z = 3(x + 2y + 3z).
$$

However, the right-hand side of the third equation is not three times the right-hand side of the first equation. The first equation says

$$
x + 2y + 3z = 6,
$$

so multiplying it by $3$ would imply

$$
3x + 6y + 9z = 18.
$$

But the system also requires

$$
3x + 6y + 9z = 12.
$$

This is a contradiction, so the system is inconsistent. Therefore, the system has

$$
\boxed{0 \text{ solutions}}.
$$


---

### 2. Asymptotic Powers of $2 \times 2$ Matrices (6 points)

For each matrix below, determine the behavior of

$$
\lim_{n \to \infty} M^n.
$$

*Hint: Use the eigen-decomposition $M = PDP^{-1}$, where $D = \{\lambda_1, \lambda_2, \dots\}$ is a diagonal matrix of the eigenvalues. What would be the formula for $M^n$?*

#### (a)

$$
M_1 =
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}.
$$

**Answer:**

$M_1$ has eigenvalues $1$ and $-1$. Also,

$$
M_1^2 =
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
= I.
$$

Therefore,

$$
M_1^n =
\begin{cases}
I, & n \text{ even},\\
M_1, & n \text{ odd}.
\end{cases}
$$

Since the sequence alternates between $I$ and $M_1$, the limit does not exist:

$$
\boxed{\lim_{n \to \infty} M_1^n \text{ does not exist}.}
$$


#### (b)

$$
M_2 =
\begin{bmatrix}
2 & -5\\
1/2 & -7/6
\end{bmatrix}.
$$

**Answer:**

The trace and determinant are

$$
\operatorname{tr}(M_2) = 2 - \frac{7}{6} = \frac{5}{6},
\qquad
\det(M_2) = 2\left(-\frac{7}{6}\right) - (-5)\left(\frac12\right)
= -\frac{7}{3} + \frac{5}{2}
= \frac16.
$$

Thus the characteristic polynomial is

$$
\lambda^2 - \frac56\lambda + \frac16 = 0.
$$

Multiplying by $6$,

$$
6\lambda^2 - 5\lambda + 1 = 0
= (3\lambda - 1)(2\lambda - 1).
$$

So the eigenvalues are

$$
\lambda_1 = \frac13,
\qquad
\lambda_2 = \frac12.
$$

Both eigenvalues have absolute value less than $1$, so

$$
\lambda_1^n \to 0,
\qquad
\lambda_2^n \to 0.
$$

Using $M_2 = PDP^{-1}$, we have

$$
M_2^n = PD^nP^{-1} \to P
\begin{bmatrix}
0 & 0\\
0 & 0
\end{bmatrix}
P^{-1}
=
\begin{bmatrix}
0 & 0\\
0 & 0
\end{bmatrix}.
$$

Therefore,

$$
\boxed{\lim_{n \to \infty} M_2^n =
\begin{bmatrix}
0 & 0\\
0 & 0
\end{bmatrix}.}
$$


#### (c)

$$
M_3 =
\begin{bmatrix}
4 & -2\\
1 & 1
\end{bmatrix}.
$$

**Answer:**

The trace and determinant are

$$
\operatorname{tr}(M_3) = 4 + 1 = 5,
\qquad
\det(M_3) = 4(1) - (-2)(1) = 6.
$$

Thus the characteristic polynomial is

$$
\lambda^2 - 5\lambda + 6 = 0
= (\lambda - 2)(\lambda - 3).
$$

So the eigenvalues are

$$
\lambda_1 = 2,
\qquad
\lambda_2 = 3.
$$

Both eigenvalues have absolute value greater than $1$, so their powers grow without bound:

$$
2^n \to \infty,
\qquad
3^n \to \infty.
$$

Since $M_3$ has two distinct eigenvalues, it is diagonalizable, and

$$
M_3^n = PD^nP^{-1}.
$$

The entries of $D^n$ grow without bound, so $M_3^n$ does not converge to a finite matrix. Therefore,

$$
\boxed{\lim_{n \to \infty} M_3^n \text{ does not exist; the powers diverge}.}
$$


---

### 3. Singular Value Decomposition (4 points)

Given the matrix

$$
A =
\begin{bmatrix}
2 & -1\\
2 & 2
\end{bmatrix},
$$

find a unit vector $x$ $(\|x\| = 1)$ for which $\|Ax\|$ is maximized.

*Hint: Recall from SVD that $A = U\Sigma V^\top$, where $A$ maps each right singular vector (a column of $V$) to a left singular vector (a column of $U$), scaled by the corresponding singular value. For which right singular vector does this scaling make $\|Ax\|$ the largest?*

**Answer:**

To maximize $\|Ax\|$ subject to $\|x\| = 1$, we use

$$
\|Ax\|^2 = x^\top A^\top A x.
$$

Thus the maximum occurs when $x$ is a unit eigenvector of $A^\top A$ corresponding to its largest eigenvalue.

First compute

$$
A^\top A
=
\begin{bmatrix}
2 & 2\\
-1 & 2
\end{bmatrix}
\begin{bmatrix}
2 & -1\\
2 & 2
\end{bmatrix}
=
\begin{bmatrix}
8 & 2\\
2 & 5
\end{bmatrix}.
$$

The characteristic polynomial is

$$
\det
\begin{bmatrix}
8-\lambda & 2\\
2 & 5-\lambda
\end{bmatrix}
= (8-\lambda)(5-\lambda)-4
= \lambda^2 - 13\lambda + 36.
$$

Factoring,

$$
\lambda^2 - 13\lambda + 36
= (\lambda - 9)(\lambda - 4).
$$

The largest eigenvalue is $\lambda = 9$. Now solve

$$
(A^\top A - 9I)x = 0.
$$

That is,

$$
\begin{bmatrix}
-1 & 2\\
2 & -4
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
= 0.
$$

The first row gives $-x_1 + 2x_2 = 0$, so $x_1 = 2x_2$. One eigenvector is therefore

$$
\begin{bmatrix}
2\\
1
\end{bmatrix}.
$$

Normalizing it gives

$$
x =
\frac{1}{\sqrt{5}}
\begin{bmatrix}
2\\
1
\end{bmatrix}.
$$

Therefore, a unit vector maximizing $\|Ax\|$ is

$$
\boxed{
x =
\begin{bmatrix}
\frac{2}{\sqrt{5}}\\
\frac{1}{\sqrt{5}}
\end{bmatrix}
}.
$$

For this vector, the maximum value is

$$
\|Ax\| = \sqrt{9} = 3.
$$


---

### 4. Image Flipping (6 points)

*This question is a warmup to problem 4 in the coding section.*

Consider a tiny $3 \times 3$ image represented by matrix $I$ and the same image flipped on the Y axis $I_{\text{flip}}$:

$$
I =
\begin{bmatrix}
x_1 & x_2 & x_3\\
x_4 & x_5 & x_6\\
x_7 & x_8 & x_9
\end{bmatrix},
\quad
I_{\text{flip}} =
\begin{bmatrix}
x_7 & x_8 & x_9\\
x_4 & x_5 & x_6\\
x_1 & x_2 & x_3
\end{bmatrix}.
$$

#### (a)

What is the size of the transformation matrix $T$ that performs this flip?

*Hint: You can first convert $I$ to a $1 \times 9$ vector, make transformation using matrix $T$, and then convert $I_{\text{flip}}$ back to a $3 \times 3$ matrix.*

**Answer:**

After flattening the $3 \times 3$ image into a $1 \times 9$ row vector, the transformation should map a $1 \times 9$ vector to another $1 \times 9$ vector. Therefore, $T$ must have size

$$
\boxed{9 \times 9}.
$$


#### (b)

Construct $T$ and verify that $I \times T$ produces $I_{\text{flip}}$.

**Answer:**

Flatten $I$ in row-major order:

$$
v =
\begin{bmatrix}
x_1 & x_2 & x_3 & x_4 & x_5 & x_6 & x_7 & x_8 & x_9
\end{bmatrix}.
$$

The flipped image should correspond to

$$
v_{\text{flip}} =
\begin{bmatrix}
x_7 & x_8 & x_9 & x_4 & x_5 & x_6 & x_1 & x_2 & x_3
\end{bmatrix}.
$$

So we want $vT = v_{\text{flip}}$. The required transformation matrix is the permutation matrix

$$
T =
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}.
$$

Then

$$
vT =
\begin{bmatrix}
x_7 & x_8 & x_9 & x_4 & x_5 & x_6 & x_1 & x_2 & x_3
\end{bmatrix}.
$$

Reshaping this row vector back into a $3 \times 3$ matrix gives

$$
I_{\text{flip}} =
\begin{bmatrix}
x_7 & x_8 & x_9\\
x_4 & x_5 & x_6\\
x_1 & x_2 & x_3
\end{bmatrix}.
$$


#### (c)

Describe an algorithm for constructing the vertical-flip transformation matrix for any $N \times N$ matrix, either text explanation or pseudocode is acceptable. Show that $I \times T$ produces $I_{\text{flip}}$.

**Answer:**

Use row-major indexing. For an $N \times N$ image, flatten the image into a row vector of length $N^2$. Let the original entry at row $r$ and column $c$ have flattened index

$$
p = rN + c,
$$

where $r,c \in \{0, 1, \dots, N-1\}$. A vertical flip swaps row $r$ with row $N-1-r$ while keeping the column the same. Thus the destination index is

$$
q = (N - 1 - r)N + c.
$$

Construct an $N^2 \times N^2$ matrix $T$ initialized to all zeros, and set

$$
T_{p,q} = 1
$$

for every pair $(r,c)$. All other entries remain zero.

In pseudocode:

```text
T = zero matrix of size N^2 x N^2

for r = 0 to N - 1:
    for c = 0 to N - 1:
        p = r * N + c
        q = (N - 1 - r) * N + c
        T[p, q] = 1
```

If $v$ is the row-major flattening of $I$, then $vT$ places each entry from row $r$ into row $N-1-r$. Therefore, reshaping $vT$ gives the vertically flipped image $I_{\text{flip}}$.


#### (d)

Describe how you would modify this algorithm to do a *horizontal* flip, along the X axis.

**Answer:**

For a horizontal flip, keep the row fixed and reverse the columns. With row-major indexing,

$$
p = rN + c
$$

is the original flattened index. The destination index should be

$$
q = rN + (N - 1 - c).
$$

So the algorithm is the same except that we set

$$
T_{p,q} = 1
$$

using $q = rN + (N - 1 - c)$.

For the $3 \times 3$ example, this would produce

$$
\begin{bmatrix}
x_3 & x_2 & x_1\\
x_6 & x_5 & x_4\\
x_9 & x_8 & x_7
\end{bmatrix}.
$$


---

## Calculus

### 5. Partial Derivatives (8 points)

First and second derivatives.

#### (a)

For $f(x_1, x_2) = x_1^3 + x_2^3 - 3x_1x_2$, find all the first and second order partial derivatives.

**Answer:**

The first-order partial derivatives are

$$
\frac{\partial f}{\partial x_1}
= 3x_1^2 - 3x_2,
\qquad
\frac{\partial f}{\partial x_2}
= 3x_2^2 - 3x_1.
$$

The second-order partial derivatives are

$$
\frac{\partial^2 f}{\partial x_1^2}
= 6x_1,
\qquad
\frac{\partial^2 f}{\partial x_2^2}
= 6x_2,
$$

and the mixed partial derivatives are

$$
\frac{\partial^2 f}{\partial x_1 \partial x_2}
= -3,
\qquad
\frac{\partial^2 f}{\partial x_2 \partial x_1}
= -3.
$$


#### (b)

Let

$$
f(x, y) = 4x^2 + y^2 - 8xy + 4x + 6y - 10.
$$

Find the critical points by solving

$$
\frac{\partial f}{\partial x} = 0
\quad \text{and} \quad
\frac{\partial f}{\partial y} = 0
$$

simultaneously and determine which point(s) yield a minimum value.

**Answer:**

First compute the partial derivatives:

$$
\frac{\partial f}{\partial x}
= 8x - 8y + 4,
\qquad
\frac{\partial f}{\partial y}
= 2y - 8x + 6.
$$

Set both equal to zero:

$$
8x - 8y + 4 = 0,
\qquad
2y - 8x + 6 = 0.
$$

Simplifying,

$$
y = x + \frac12,
\qquad
y = 4x - 3.
$$

Therefore,

$$
x + \frac12 = 4x - 3
\quad \Longrightarrow \quad
\frac72 = 3x
\quad \Longrightarrow \quad
x = \frac76.
$$

Then

$$
y = x + \frac12 = \frac76 + \frac12 = \frac53.
$$

So the only critical point is

$$
\left(\frac76, \frac53\right).
$$

To determine whether it is a minimum, consider the Hessian:

$$
H =
\begin{bmatrix}
8 & -8\\
-8 & 2
\end{bmatrix}.
$$

Its determinant is

$$
\det(H) = 8 \cdot 2 - (-8)^2 = 16 - 64 = -48 < 0.
$$

Since the Hessian is indefinite, the critical point is a saddle point, not a minimum. Thus,

$$
\boxed{\text{there are no critical points that yield a minimum value}.}
$$


#### (c)

For $f(x, y) = e^{xy} + x^2y$, compute:

##### (i)

$$
\frac{\partial f}{\partial x}
\quad \text{and} \quad
\frac{\partial f}{\partial y}.
$$

**Answer:**

Using the product rule and chain rule,

$$
\frac{\partial f}{\partial x}
= y e^{xy} + 2xy,
$$

and

$$
\frac{\partial f}{\partial y}
= x e^{xy} + x^2.
$$


##### (ii)

$$
\frac{\partial^2 f}{\partial x^2},
\quad
\frac{\partial^2 f}{\partial y^2},
\quad \text{and} \quad
\frac{\partial^2 f}{\partial x \partial y}.
$$

**Answer:**

The second-order partial derivatives are

$$
\frac{\partial^2 f}{\partial x^2}
= y^2 e^{xy} + 2y,
$$

$$
\frac{\partial^2 f}{\partial y^2}
= x^2 e^{xy},
$$

and

$$
\frac{\partial^2 f}{\partial x \partial y}
= \frac{\partial}{\partial y}
\left(y e^{xy} + 2xy\right)
= e^{xy} + xy e^{xy} + 2x.
$$

So

$$
\frac{\partial^2 f}{\partial x \partial y}
= (1 + xy)e^{xy} + 2x.
$$


#### (d)

For $f(x, y) = \ln(x^2 + y^2 + 1)$, compute:

##### (i)

$$
\frac{\partial f}{\partial x}
\quad \text{and} \quad
\frac{\partial f}{\partial y}.
$$

**Answer:**

Let

$$
s = x^2 + y^2 + 1.
$$

Then $f(x,y) = \ln(s)$. By the chain rule,

$$
\frac{\partial f}{\partial x}
= \frac{1}{s}(2x)
= \frac{2x}{x^2 + y^2 + 1},
$$

and

$$
\frac{\partial f}{\partial y}
= \frac{1}{s}(2y)
= \frac{2y}{x^2 + y^2 + 1}.
$$


##### (ii)

$$
\frac{\partial^2 f}{\partial x^2},
\quad
\frac{\partial^2 f}{\partial y^2},
\quad \text{and} \quad
\frac{\partial^2 f}{\partial x \partial y}.
$$

**Answer:**

Using $s = x^2 + y^2 + 1$, we have

$$
\frac{\partial^2 f}{\partial x^2}
= \frac{\partial}{\partial x}\left(\frac{2x}{s}\right)
= \frac{2s - 4x^2}{s^2}.
$$

Therefore,

$$
\frac{\partial^2 f}{\partial x^2}
= \frac{2(1 - x^2 + y^2)}{(x^2 + y^2 + 1)^2}.
$$

Similarly,

$$
\frac{\partial^2 f}{\partial y^2}
= \frac{2s - 4y^2}{s^2}
= \frac{2(1 + x^2 - y^2)}{(x^2 + y^2 + 1)^2}.
$$

For the mixed partial derivative,

$$
\frac{\partial^2 f}{\partial x \partial y}
= \frac{\partial}{\partial y}\left(\frac{2x}{s}\right)
= -\frac{4xy}{s^2}.
$$

Thus,

$$
\boxed{
\frac{\partial^2 f}{\partial x \partial y}
= -\frac{4xy}{(x^2 + y^2 + 1)^2}
}.
$$


---

### 6. Recursive Expression and Derivatives (6 points)

Suppose we have variables $z_1, z_2, \dots, z_n$, $w_1, \dots, w_{n-1}$, and $b_1, \dots, b_{n-1} \in \mathbb{R}$, where

$$
z_n = w_{n-1}z_{n-1} + b_{n-1}.
$$

#### (a)

Compute

$$
\frac{dz_k}{dz_{k-1}}
$$

for $k = 2, \dots, n$.

**Answer:**

For each $k=2,\dots,n$, the recurrence is

$$
z_k=w_{k-1}z_{k-1}+b_{k-1}.
$$

Treating $w_{k-1}$ and $b_{k-1}$ as constants with respect to
$z_{k-1}$, we obtain

$$
\boxed{\frac{dz_k}{dz_{k-1}}=w_{k-1}}.
$$

#### (b)

Compute

$$
\frac{dz_n}{dz_1}.
$$

**Answer:**

By repeatedly applying the chain rule,

$$
\frac{dz_n}{dz_1}
=
\frac{dz_n}{dz_{n-1}}
\frac{dz_{n-1}}{dz_{n-2}}
\cdots
\frac{dz_2}{dz_1}.
$$

Using the result from part (a), this becomes

$$
\boxed{
\frac{dz_n}{dz_1}
=w_{n-1}w_{n-2}\cdots w_1
=\prod_{i=1}^{n-1}w_i
}.
$$

#### (c)

Compute

$$
\frac{dz_n}{db_1}.
$$

You may express your answers in terms of $z_1, \dots, z_n, w_1, \dots, w_{n-1}, b_1, \dots, b_{n-1}$.

**Answer:**

First,

$$
z_2=w_1z_1+b_1,
$$

so, treating $z_1$ and $w_1$ as independent of $b_1$,

$$
\frac{dz_2}{db_1}=1.
$$

Applying the chain rule from $z_2$ through $z_n$,

$$
\frac{dz_n}{db_1}
=
\frac{dz_n}{dz_{n-1}}
\frac{dz_{n-1}}{dz_{n-2}}
\cdots
\frac{dz_3}{dz_2}
\frac{dz_2}{db_1}.
$$

Therefore,

$$
\boxed{
\frac{dz_n}{db_1}
=w_{n-1}w_{n-2}\cdots w_2
=\prod_{i=2}^{n-1}w_i
}.
$$

When $n=2$, the product is empty and is defined to equal $1$, which
agrees with $\frac{dz_2}{db_1}=1$.

---

## Probability

### 7. Conditioned Uniform Difference (3 points)

Let $X, Y \stackrel{\text{iid}}{\sim} \text{Unif}(-1, 1)$. Compute

$$
\mathbb{P}\left(|X - Y| \le 0.5 \mid XY > 0\right).
$$

*Hint: Try drawing a 2D cartesian plane where the horizontal and vertical axes represent $X$ and $Y$ respectively and each range from -1 to 1. For which quadrants is it true that $XY > 0$? Within these quadrants, how can we visualize the region $|X - Y| < 0.5$?*

**Answer:**

Because $X$ and $Y$ are uniformly distributed on $(-1,1)$, probabilities
can be computed as ratios of areas in the square $(-1,1)^2$.

The condition $XY>0$ means that $X$ and $Y$ have the same sign. Thus,
the conditional region consists of the two unit squares

$$
(0,1)\times(0,1)
\quad\text{and}\quad
(-1,0)\times(-1,0),
$$

whose total area is

$$
2.
$$

Consider the square $(0,1)\times(0,1)$. The event
$|X-Y|\leq 0.5$ is the region between the lines

$$
y=x-0.5
\quad\text{and}\quad
y=x+0.5.
$$

The excluded region consists of two right triangles, each with legs of
length $0.5$. Therefore, the area satisfying $|X-Y|\leq 0.5$ in this
unit square is

$$
1-2\left(\frac12\right)(0.5)^2
=1-\frac14
=\frac34.
$$

By symmetry, the satisfying area in $(-1,0)\times(-1,0)$ is also
$\frac34$. Hence,

$$
\mathbb{P}\left(|X-Y|\leq 0.5\mid XY>0\right)
=
\frac{\frac34+\frac34}{2}
=\boxed{\frac34}.
$$

---

### 8. Nearest-Neighbor Arc Length (4 points)

Suppose you select $20$ i.i.d. points $X_1, \dots, X_{20}$ uniformly at random on the circumference of the unit circle.

#### (a)

Let $D$ be the shortest arc distance from $X_1$ to the nearest of the other 19 points. Calculate $\mathbb{P}(D > t)$ where $0 \le t \le \frac{1}{2}$.

*Hint: What does the event $\{D > t\}$ mean in terms of where the other 19 points can be located relative to $X_1$?*

**Answer:**

Measure arc length as a fraction of the full circumference, so the
circle has total length $1$. Fix $X_1$. The event $D>t$ means that none
of the other $19$ points is within distance $t$ clockwise or
counterclockwise from $X_1$.

The two excluded arcs have total length $2t$. Thus, each of the other
$19$ points has probability $1-2t$ of lying outside these arcs. Since
the points are independent,

$$
\boxed{
\mathbb{P}(D>t)=(1-2t)^{19},
\qquad 0\leq t\leq \frac12
}.
$$

#### (b)

Find the expected arc length, in degrees, between $X_1$ and the point nearest to it.

*Hint: The Wikipedia page on expected value of continuous variables may be helpful here.*

**Answer:**

Using the tail-integral formula,

$$
\mathbb{E}[D]
=\int_0^\infty \mathbb{P}(D>t)\,dt.
$$

Because $0\leq D\leq \frac12$, part (a) gives

$$
\begin{aligned}
\mathbb{E}[D]
&=\int_0^{1/2}(1-2t)^{19}\,dt\\
&=\frac12\int_0^1 u^{19}\,du\\
&=\frac12\cdot\frac1{20}\\
&=\frac1{40},
\end{aligned}
$$

where $u=1-2t$. Hence the expected distance is $\frac1{40}$ of the
full circle. Converting to degrees,

$$
\boxed{
\mathbb{E}[D]
=360^\circ\cdot\frac1{40}
=9^\circ
}.
$$

---

### 9. Cancer Screening (3 points)

A medical test has sensitivity $90\%$ and false-positive rate $3\%$:

$$
\mathbb{P}(T = 1 \mid C = 1) = 0.9,
\qquad
\mathbb{P}(T = 1 \mid C = 0) = 0.03.
$$

Suppose the disease prevalence is very low, $\mathbb{P}(C = 1) = 0.001$. Compute the posterior probability of disease given a positive test:

$$
\mathbb{P}(C = 1 \mid T = 1).
$$

**Answer:**

By Bayes' rule,

$$
\mathbb{P}(C=1\mid T=1)
=
\frac{
\mathbb{P}(T=1\mid C=1)\mathbb{P}(C=1)
}{
\mathbb{P}(T=1)
}.
$$

The probability of a positive test is

$$
\begin{aligned}
\mathbb{P}(T=1)
&=\mathbb{P}(T=1\mid C=1)\mathbb{P}(C=1)\\
&\quad+\mathbb{P}(T=1\mid C=0)\mathbb{P}(C=0)\\
&=(0.9)(0.001)+(0.03)(0.999)\\
&=0.0009+0.02997\\
&=0.03087.
\end{aligned}
$$

Therefore,

$$
\begin{aligned}
\mathbb{P}(C=1\mid T=1)
&=\frac{(0.9)(0.001)}{0.03087}\\
&=\frac{0.0009}{0.03087}\\
&\approx 0.02915.
\end{aligned}
$$

Thus,

$$
\boxed{
\mathbb{P}(C=1\mid T=1)\approx 2.92\%
}.
$$

---

### 10. Follower Counts (3 points)

Suppose $420$ people are sitting uniformly at random around a circle, each with a *distinct* number of TikTok followers. What is the expected number of people whose follower count is higher than both their immediate neighbors?

**Answer:**

For each person $i$, define the indicator variable

$$
I_i=
\begin{cases}
1, & \text{if person }i\text{ has more followers than both neighbors},\\
0, & \text{otherwise}.
\end{cases}
$$

Among person $i$ and their two immediate neighbors, each of the three
people is equally likely to have the largest follower count. Since all
follower counts are distinct,

$$
\mathbb{P}(I_i=1)=\frac13.
$$

Let

$$
N=\sum_{i=1}^{420}I_i
$$

be the total number of people whose follower count exceeds both
neighbors. By linearity of expectation,

$$
\begin{aligned}
\mathbb{E}[N]
&=\sum_{i=1}^{420}\mathbb{E}[I_i]\\
&=\sum_{i=1}^{420}\mathbb{P}(I_i=1)\\
&=420\left(\frac13\right)\\
&=\boxed{140}.
\end{aligned}
$$

The indicator variables do not need to be independent for linearity of
expectation to apply.
