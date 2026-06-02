---
title: "ADMM 算法系统讲义"
author: "Notes for Computer Science Students"
date: "2026-05-30"
lang: zh-CN
documentclass: ctexart
geometry: margin=1in
fontsize: 11pt
toc: true
toc-depth: 3
numbersections: true
header-includes:
  - \usepackage{amsmath, amssymb, bm, mathtools}
  - \usepackage{algorithm}
  - \usepackage{algpseudocode}
  - \usepackage{booktabs}
  - \usepackage{hyperref}
---

# 学习目标

ADMM，全称为 Alternating Direction Method of Multipliers，即交替方向乘子法，是一种求解带约束凸优化问题的算法。它把一个复杂问题拆成几个更容易处理的子问题，然后通过拉格朗日乘子协调这些子问题，使它们逐步满足原始约束。

读完本文后，你应该能够理解：

1. ADMM 想解决什么类型的优化问题。
2. ADMM 和拉格朗日乘子法、增广拉格朗日法之间的关系。
3. 标准 ADMM 的数学形式和迭代流程。
4. 为什么 ADMM 适合处理可分解目标函数。
5. 如何从一个具体优化问题写出 ADMM 更新步骤。
6. ADMM 的收敛条件、停止准则和实现注意事项。

# 优化问题的基本背景

很多机器学习、信号处理、图优化和分布式计算问题都可以写成如下形式：

\[
\min_x f(x)
\]

其中 \(x\) 是待优化变量，\(f(x)\) 是目标函数。

如果问题还带有约束，例如：

\[
\min_x f(x)
\quad \text{s.t.} \quad Ax=b
\]

我们就需要在优化目标的同时保证约束成立。

ADMM 最常处理的标准形式是：

\[
\min_{x,z} f(x)+g(z)
\quad \text{s.t.} \quad Ax+Bz=c
\]

其中：

- \(x\) 和 \(z\) 是两组变量；
- \(f\) 和 \(g\) 是两个目标项；
- \(Ax+Bz=c\) 是线性等式约束；
- \(A,B,c\) 是给定矩阵或向量。

这种形式很有用，因为很多问题的目标函数可以自然拆成两部分。例如：

\[
\min_x \ell(x)+\lambda R(x)
\]

其中 \(\ell(x)\) 是损失函数，\(R(x)\) 是正则项。若 \(\ell\) 容易做梯度更新，而 \(R\) 容易做近端映射，那么 ADMM 就可以把它们分开处理。

# 从拉格朗日乘子法到 ADMM

## 普通拉格朗日乘子法

考虑等式约束问题：

\[
\min_x f(x)
\quad \text{s.t.} \quad Ax=b
\]

普通拉格朗日函数为：

\[
L(x,y)=f(x)+y^\top(Ax-b)
\]

其中 \(y\) 是拉格朗日乘子，也叫对偶变量。

直觉上，\(y^\top(Ax-b)\) 用来惩罚违反约束的解。如果 \(Ax-b\neq 0\)，对偶变量会调整惩罚方向，使算法逐渐推向可行区域。

## 增广拉格朗日法

普通拉格朗日函数只使用线性惩罚项，有时数值稳定性较差。增广拉格朗日法在其中加入二次惩罚项：

\[
L_\rho(x,y)
=
f(x)+y^\top(Ax-b)+\frac{\rho}{2}\|Ax-b\|_2^2
\]

其中 \(\rho>0\) 是惩罚参数。

增广拉格朗日法的迭代通常为：

\[
x^{k+1}
=
\arg\min_x L_\rho(x,y^k)
\]

\[
y^{k+1}
=
y^k+\rho(Ax^{k+1}-b)
\]

这里的核心思想是：

- 先固定对偶变量 \(y^k\)，优化原始变量 \(x\)；
- 再根据约束残差 \(Ax^{k+1}-b\) 更新对偶变量；
- 如果约束违反严重，对偶变量会增加惩罚力度。

## ADMM 的出现

如果问题是：

\[
\min_{x,z} f(x)+g(z)
\quad \text{s.t.} \quad Ax+Bz=c
\]

对应的增广拉格朗日函数是：

\[
L_\rho(x,z,y)
=
f(x)+g(z)
+y^\top(Ax+Bz-c)
+\frac{\rho}{2}\|Ax+Bz-c\|_2^2
\]

如果直接同时最小化 \(x,z\)，可能仍然很难：

\[
(x^{k+1},z^{k+1})
=
\arg\min_{x,z} L_\rho(x,z,y^k)
\]

ADMM 的关键做法是：不要同时更新 \(x,z\)，而是交替更新。

# 标准 ADMM 算法

标准 ADMM 解决：

\[
\min_{x,z} f(x)+g(z)
\quad \text{s.t.} \quad Ax+Bz=c
\]

增广拉格朗日函数为：

\[
L_\rho(x,z,y)
=
f(x)+g(z)
+y^\top(Ax+Bz-c)
+\frac{\rho}{2}\|Ax+Bz-c\|_2^2
\]

ADMM 的更新为：

\[
x^{k+1}
=
\arg\min_x
L_\rho(x,z^k,y^k)
\]

\[
z^{k+1}
=
\arg\min_z
L_\rho(x^{k+1},z,y^k)
\]

\[
y^{k+1}
=
y^k+\rho(Ax^{k+1}+Bz^{k+1}-c)
\]

也就是说：

1. 固定 \(z^k,y^k\)，更新 \(x\)；
2. 固定新的 \(x^{k+1}\) 和旧的 \(y^k\)，更新 \(z\)；
3. 根据约束残差更新对偶变量 \(y\)。

## 缩放形式

在实现中，更常用 scaled form。定义：

\[
u=\frac{1}{\rho}y
\]

则增广拉格朗日中的二次项可以改写为：

\[
y^\top r+\frac{\rho}{2}\|r\|_2^2
=
\frac{\rho}{2}\|r+u\|_2^2
-\frac{\rho}{2}\|u\|_2^2
\]

其中：

\[
r=Ax+Bz-c
\]

忽略与 \(x,z\) 无关的常数项后，ADMM 可写成：

\[
x^{k+1}
=
\arg\min_x
f(x)+\frac{\rho}{2}
\|Ax+Bz^k-c+u^k\|_2^2
\]

\[
z^{k+1}
=
\arg\min_z
g(z)+\frac{\rho}{2}
\|Ax^{k+1}+Bz-c+u^k\|_2^2
\]

\[
u^{k+1}
=
u^k+Ax^{k+1}+Bz^{k+1}-c
\]

这是工程实现里最常见的 ADMM 形式。

# ADMM 的伪代码

\[
\begin{aligned}
&\textbf{Input: } f,g,A,B,c,\rho>0,\ x^0,z^0,u^0 \\
&\textbf{for } k=0,1,2,\dots \\
&\quad x^{k+1}
=
\arg\min_x
f(x)+\frac{\rho}{2}
\|Ax+Bz^k-c+u^k\|_2^2 \\
&\quad z^{k+1}
=
\arg\min_z
g(z)+\frac{\rho}{2}
\|Ax^{k+1}+Bz-c+u^k\|_2^2 \\
&\quad u^{k+1}
=
u^k+Ax^{k+1}+Bz^{k+1}-c \\
&\quad \text{check convergence} \\
&\textbf{end for}
\end{aligned}
\]

在程序中，最重要的是能否高效求解两个子问题：

\[
x^{k+1}=\arg\min_x(\cdots)
\]

\[
z^{k+1}=\arg\min_z(\cdots)
\]

如果这两个子问题都有闭式解，ADMM 通常非常高效。如果没有闭式解，也可以在子问题内部使用梯度下降、牛顿法或共轭梯度法近似求解。

# Consensus ADMM

很多问题可以写成：

\[
\min_x f(x)+g(x)
\]

但是 \(f\) 和 \(g\) 各自容易处理，合在一起不好处理。此时可以引入辅助变量 \(z\)，写成：

\[
\min_{x,z} f(x)+g(z)
\quad \text{s.t.} \quad x=z
\]

这就是 consensus 形式。它对应：

\[
A=I,\quad B=-I,\quad c=0
\]

scaled ADMM 为：

\[
x^{k+1}
=
\arg\min_x
f(x)+\frac{\rho}{2}\|x-z^k+u^k\|_2^2
\]

\[
z^{k+1}
=
\arg\min_z
g(z)+\frac{\rho}{2}\|x^{k+1}-z+u^k\|_2^2
\]

\[
u^{k+1}=u^k+x^{k+1}-z^{k+1}
\]

这个形式在机器学习中非常常见。例如：

- \(f\) 是数据拟合项；
- \(g\) 是稀疏正则项、非负约束、盒约束、单纯形约束等。

# 近端算子视角

近端算子定义为：

\[
\mathrm{prox}_{\lambda g}(v)
=
\arg\min_z
g(z)+\frac{1}{2\lambda}\|z-v\|_2^2
\]

在 consensus ADMM 中，\(z\)-update 是：

\[
z^{k+1}
=
\arg\min_z
g(z)+\frac{\rho}{2}
\|z-(x^{k+1}+u^k)\|_2^2
\]

所以：

\[
z^{k+1}
=
\mathrm{prox}_{g/\rho}(x^{k+1}+u^k)
\]

这说明 ADMM 和近端优化方法关系很密切。只要某个函数的 prox 容易计算，它就适合作为 ADMM 中的一块。

# 例子一：Lasso 问题

Lasso 是经典稀疏回归问题：

\[
\min_x
\frac{1}{2}\|Ax-b\|_2^2+\lambda\|x\|_1
\]

其中 \(\|x\|_1\) 会鼓励解变稀疏。

引入 \(z\)，令 \(x=z\)：

\[
\min_{x,z}
\frac{1}{2}\|Ax-b\|_2^2+\lambda\|z\|_1
\quad \text{s.t.} \quad x=z
\]

scaled ADMM 为：

\[
x^{k+1}
=
\arg\min_x
\frac{1}{2}\|Ax-b\|_2^2
+\frac{\rho}{2}\|x-z^k+u^k\|_2^2
\]

这是一个二次优化问题，闭式解为：

\[
x^{k+1}
=
(A^\top A+\rho I)^{-1}
\left(A^\top b+\rho(z^k-u^k)\right)
\]

\[
z^{k+1}
=
\arg\min_z
\lambda\|z\|_1
+\frac{\rho}{2}\|x^{k+1}-z+u^k\|_2^2
\]

这等价于 soft-thresholding：

\[
z^{k+1}
=
S_{\lambda/\rho}(x^{k+1}+u^k)
\]

其中：

\[
S_\kappa(a)
=
\mathrm{sign}(a)\max(|a|-\kappa,0)
\]

最后更新：

\[
u^{k+1}=u^k+x^{k+1}-z^{k+1}
\]

# 例子二：带非负约束的最小二乘

考虑：

\[
\min_x \frac{1}{2}\|Ax-b\|_2^2
\quad \text{s.t.} \quad x\ge 0
\]

把约束写成指示函数：

\[
I_{\mathbb R_+^n}(z)
=
\begin{cases}
0, & z\ge 0 \\
+\infty, & \text{otherwise}
\end{cases}
\]

问题变成：

\[
\min_{x,z}
\frac{1}{2}\|Ax-b\|_2^2
+I_{\mathbb R_+^n}(z)
\quad \text{s.t.} \quad x=z
\]

ADMM 更新为：

\[
x^{k+1}
=
(A^\top A+\rho I)^{-1}
\left(A^\top b+\rho(z^k-u^k)\right)
\]

\[
z^{k+1}
=
\Pi_{\mathbb R_+^n}(x^{k+1}+u^k)
=
\max(x^{k+1}+u^k,0)
\]

\[
u^{k+1}=u^k+x^{k+1}-z^{k+1}
\]

这里的 \(z\)-update 就是投影到非负正交象限。

# 例子三：矩阵投影问题

假设给定矩阵 \(M\in\mathbb R^{m\times n}\)，希望找到离它最近的矩阵 \(X\)，并满足：

\[
X_{ij}\ge 0,\quad
X\mathbf 1\le \mathbf 1,\quad
X^\top\mathbf 1\le \mathbf 1
\]

也就是元素非负，每一行和每一列的和都不超过 \(1\)。欧氏投影问题是：

\[
\min_X \frac{1}{2}\|X-M\|_F^2
\quad \text{s.t.} \quad
X_{ij}\ge 0,\
X\mathbf 1\le \mathbf 1,\
X^\top\mathbf 1\le \mathbf 1
\]

定义两个集合：

\[
\mathcal R
=
\{Y:Y\ge 0,\ Y\mathbf 1\le \mathbf 1\}
\]

\[
\mathcal C
=
\{Z:Z\ge 0,\ Z^\top\mathbf 1\le \mathbf 1\}
\]

引入 \(Y,Z\)，令：

\[
X=Y,\quad X=Z
\]

得到：

\[
\min_{X,Y,Z}
\frac{1}{2}\|X-M\|_F^2
+I_{\mathcal R}(Y)
+I_{\mathcal C}(Z)
\quad
\text{s.t.}
\quad X=Y,\ X=Z
\]

scaled ADMM 为：

\[
X^{k+1}
=
\frac{
M+\rho(Y^k-U^k)+\rho(Z^k-V^k)
}{
1+2\rho
}
\]

\[
Y^{k+1}
=
\Pi_{\mathcal R}(X^{k+1}+U^k)
\]

\[
Z^{k+1}
=
\Pi_{\mathcal C}(X^{k+1}+V^k)
\]

\[
U^{k+1}=U^k+X^{k+1}-Y^{k+1}
\]

\[
V^{k+1}=V^k+X^{k+1}-Z^{k+1}
\]

其中 \(\Pi_{\mathcal R}\) 是逐行投影到集合：

\[
\{r:r\ge 0,\ \sum_i r_i\le 1\}
\]

\(\Pi_{\mathcal C}\) 是逐列做同样的投影。

# 原始残差与对偶残差

ADMM 通常通过两个残差判断是否收敛。

对标准形式：

\[
\min_{x,z} f(x)+g(z)
\quad \text{s.t.} \quad Ax+Bz=c
\]

原始残差为：

\[
r^{k+1}=Ax^{k+1}+Bz^{k+1}-c
\]

它衡量约束是否满足。

对偶残差为：

\[
s^{k+1}
=
\rho A^\top B(z^{k+1}-z^k)
\]

它衡量对偶变量和子问题最优性是否稳定。

常见停止准则是：

\[
\|r^{k+1}\|_2\le \epsilon_{\mathrm{pri}}
\]

\[
\|s^{k+1}\|_2\le \epsilon_{\mathrm{dual}}
\]

其中：

\[
\epsilon_{\mathrm{pri}}
=
\sqrt{p}\epsilon_{\mathrm{abs}}
+\epsilon_{\mathrm{rel}}
\max\{\|Ax^{k+1}\|_2,\|Bz^{k+1}\|_2,\|c\|_2\}
\]

\[
\epsilon_{\mathrm{dual}}
=
\sqrt{n}\epsilon_{\mathrm{abs}}
+\epsilon_{\mathrm{rel}}\|A^\top y^{k+1}\|_2
\]

这里 \(p\) 是约束维度，\(n\) 是变量维度。工程上常用：

\[
\epsilon_{\mathrm{abs}}=10^{-4},
\quad
\epsilon_{\mathrm{rel}}=10^{-3}
\]

# 惩罚参数 \(\rho\)

\(\rho\) 控制约束惩罚强度。它不会改变原问题的最优解，但会强烈影响收敛速度。

如果 \(\rho\) 太小：

- 约束惩罚弱；
- 原始残差下降慢；
- \(x,z\) 可能长时间不一致。

如果 \(\rho\) 太大：

- 约束被过分强调；
- 每步更新可能过于保守；
- 对偶残差可能下降慢。

一种常见经验策略是根据残差自适应调整：

\[
\rho^{k+1}
=
\begin{cases}
\tau\rho^k, & \|r^k\|_2>\mu\|s^k\|_2 \\
\rho^k/\tau, & \|s^k\|_2>\mu\|r^k\|_2 \\
\rho^k, & \text{otherwise}
\end{cases}
\]

常用参数：

\[
\mu=10,\quad \tau=2
\]

如果改变了 \(\rho\)，scaled dual variable \(u\) 也应相应缩放，以保持 \(y=\rho u\) 一致。

# 收敛条件

在经典凸优化设定下，ADMM 通常要求：

1. \(f\) 和 \(g\) 是闭、适当、凸函数；
2. 约束 \(Ax+Bz=c\) 存在可行解；
3. 增广拉格朗日函数的子问题可以被精确求解，或误差可控；
4. 某些技术条件成立，例如相关矩阵满足适当的秩条件。

在这些条件下，ADMM 可以保证：

\[
r^k\to 0
\]

并且目标值收敛到最优值。

如果问题非凸，ADMM 仍然常被使用，但理论保证弱得多。此时它更像一种有效的启发式算法，需要更多实验验证。

# ADMM 适合什么问题

ADMM 特别适合以下结构：

1. 目标函数可以拆成几部分；
2. 每一部分单独优化比较容易；
3. 约束可以通过投影或近端算子处理；
4. 变量规模大，但结构稀疏或可并行；
5. 希望把全局问题拆成多个局部子问题。

典型应用包括：

- Lasso 和稀疏回归；
- total variation denoising；
- matrix completion；
- robust PCA；
- constrained least squares；
- distributed optimization；
- graph learning；
- optimal transport 的某些变体；
- 神经网络中的约束投影或结构化正则。

# 实现 ADMM 的步骤

面对一个新问题，可以按以下步骤推导：

1. 写出原始优化问题。
2. 判断目标函数能否拆成 \(f(x)+g(z)\)。
3. 引入辅助变量，把问题改写成线性约束形式。
4. 写出增广拉格朗日函数。
5. 选择 scaled 或 unscaled 形式。
6. 分别推导 \(x\)-update 和 \(z\)-update。
7. 推导 dual update。
8. 写出原始残差和对偶残差。
9. 选择 \(\rho\)、停止准则和初始化方式。
10. 检查每个子问题是否有闭式解；如果没有，选择数值方法。

# 常见错误

## 错误一：变量拆分后忘记约束

例如把：

\[
\min_x f(x)+g(x)
\]

拆成：

\[
\min_{x,z} f(x)+g(z)
\]

但忘记加：

\[
x=z
\]

这样就改变了原问题。

## 错误二：符号不一致

有些资料使用：

\[
Ax+Bz=c
\]

有些使用：

\[
Ax+Bz-c=0
\]

还有些 consensus ADMM 写成：

\[
x-z=0
\]

或：

\[
z-x=0
\]

这些写法都可以，但更新公式中的符号必须保持一致。

## 错误三：误以为 ADMM 一定比梯度下降快

ADMM 的优势来自结构分解。如果子问题本身很难，或者没有可利用的 prox / projection，ADMM 未必更快。

## 错误四：忽略 \(\rho\) 的影响

理论上 \(\rho\) 不改变最优解，但在有限迭代中，它可能决定算法是否实用。

# 和其他方法的关系

ADMM 和以下方法关系密切：

- 拉格朗日乘子法：ADMM 使用对偶变量处理约束；
- 增广拉格朗日法：ADMM 是增广拉格朗日法的交替最小化版本；
- proximal gradient：ADMM 子问题经常可解释为 prox；
- Douglas--Rachford splitting：ADMM 可以看作对偶问题上的 Douglas--Rachford splitting；
- projected gradient：当约束投影容易时，两者都可用于约束优化，但 ADMM 更适合拆分复杂约束。

# 一个最小 Python 实现框架

下面是 consensus ADMM 的代码骨架：

```python
def admm(x0, z0, u0, rho, max_iter, tol):
    x = x0.copy()
    z = z0.copy()
    u = u0.copy()

    for k in range(max_iter):
        x_old = x.copy()
        z_old = z.copy()

        # x-update: solve f(x) + rho / 2 * ||x - z + u||^2
        x = update_x(z, u, rho)

        # z-update: solve g(z) + rho / 2 * ||x - z + u||^2
        z = update_z(x, u, rho)

        # dual update
        u = u + x - z

        # residuals
        r = x - z
        s = rho * (z - z_old)

        if norm(r) <= tol and norm(s) <= tol:
            break

    return x, z, u
```

实际问题中，你需要具体实现 `update_x` 和 `update_z`。

# 推荐学习路径

建议按以下顺序学习：

1. 复习凸优化基本概念：凸函数、凸集合、KKT 条件。
2. 学习拉格朗日乘子法和对偶问题。
3. 理解增广拉格朗日法。
4. 掌握标准 ADMM 和 scaled ADMM。
5. 手推 Lasso 的 ADMM 更新。
6. 手推一个投影问题的 ADMM 更新。
7. 实现一个小规模 Lasso 或 constrained least squares。
8. 阅读 Boyd 等人的 ADMM 综述。

# 参考资料

1. Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, Jonathan Eckstein. *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*. Foundations and Trends in Machine Learning, 2011.
2. Dimitri P. Bertsekas. *Nonlinear Programming*. Athena Scientific.
3. Neal Parikh, Stephen Boyd. *Proximal Algorithms*. Foundations and Trends in Optimization, 2014.

# 编译为 PDF

如果安装了 Pandoc 和 TeX Live，可以在当前目录运行：

```bash
pandoc admm_tutorial.md -o admm_tutorial.pdf --pdf-engine=xelatex
```

如果中文字体报错，可以先确认系统中安装了 CTeX 或 TeX Live 的中文支持包。本文使用 `ctexart` 文档类，通常在完整 TeX Live 环境下可以直接编译。
