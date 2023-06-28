# 凸优化的概念总结
## theory
#### 二次型
* 定义：数域 K 上的 n 元二次型是数域 K 上含有 n 个变量的二次多项式 
### convex function
#### convex function
* 定义：对于函数 $f: \mathbf{R}^n \rightarrow \mathbf{R}$，如果
  * dom f 是凸集
  * 对任意 $x,y \in dom f$ 和任意 $0 \leq \theta \leq 1$，有：$f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta) f(y)$
  则称该函数为凸函数

#### affine functions
* definition:
  * a function $f: \mathrm{R} ^n\rightarrow \mathrm{R}^m$ is affine if it is a sum of a linear function and a constant
  * i.e. f is affine if $f(x) = Ax + b$, where $A\in \mathrm{R} ^{m*n}$,
### convex optimization problems
#### optimization problems
* 定义：
    在所有满足 $f_i(x) \leqslant 0, i=1, \cdots, m$ 及 $h_i(x)=0, i=1, \cdots, p$ 的 $x$ 中寻找极小化 $f_0(x)$ 的 $x$ 的问题
    $$
    \begin{array}{ll}
    \text { minimize } & f_0(x) \\
    \text { subject to } & f_i(x) \leqslant 0, \quad i=1, \cdots, m \\
    & h_i(x)=0, \quad i=1, \cdots, p
    \end{array}
    $$
    
    * 优化变量：$x \in \mathbf{R}^n$
    * 目标函数：$f_0: \mathbf{R}^n \rightarrow \mathbf{R}$ 
    * 定义域：
    $$
    \mathcal{D}=\bigcap_{i=0}^m \operatorname{dom} f_i \cap \bigcap_{i=1}^p \operatorname{dom} h_i,
    $$


#### convex optimization problems
* 定义：对于某优化问题，如果满足：
    $$
    \begin{array}{ll}
    \text { minimize } & f_0(x) \\
    \text { subject to } & f_i(x) \leqslant 0, \quad i=1, \cdots, m \\
    & a_i^T x=b_i, \quad i=1, \cdots, p
    \end{array}
    $$
    
    * 目标的数 $f_0(x)$ 是凸函数
    * 不等式约東函数是 $f_1, \cdots, f_m$ 是凸函数
    * 等式约束 $h_1, \cdots, h_m$ 是仿射函数
    
    则称其为凸优化问题

* 注意到性质：
  * 凸优化问题的可行集是凸的, 因为它是问题定义域
    $$
    \mathcal{D}=\bigcap_{i=0}^m \operatorname{dom} f_i
    $$
    (这是一个凸集) 、 $m$ 个 (凸的) 下水平集 $\left\{x \mid f_i(x) \leqslant 0\right\}$ 以及 $p$ 个超平面 $\left\{x \mid a_i^T x=b_i\right\}$ 的交集
    
#### Linear optimization problem
* 定义：
  * 对于一个优化问题，如果其目标函数和约東函数都是仿射时, 问题称作线性规划 (LP)
  * 形式化表达：
    $$
    \begin{array}{ll}
    \text { minimize } & c^T x+d \\
    \text { subject to } & G x \preceq h \\
    & A x=b,
    \end{array}
    $$
    其中 $G \in \mathbf{R}^{m \times n}, A \in \mathbf{R}^{p \times n}$ 。
  * 线性规划是凸优化问题。

#### quadratic optimization problem
* 定义：
  * 目标函数是二次型并且约束函数为仿射时, 该问题称为二次规划 $(\mathrm{QP})$ 。
  * 形式化表达：
    $$
    \begin{array}{ll}
    \operatorname{minimize} & (1 / 2) x^T P x+q^T x+r \\
    \text { subject to } & G x \preceq h \\
    & A x=b
    \end{array}
    $$

    其中： $P \in \mathbf{S}_{+}^n, G \in \mathbf{R}^{m \times n}$ 且 $A \in \mathbf{R}^{p \times n}$
* 性质：
  * 如果Q是半正定矩阵，那么f(x)是一个凸函数。
  * 如果Q是正定矩阵，那么全局最小值就是唯一的。
  * 如果Q=0，二次规划问题就变成线性规划问题。
  * 如果有至少一个向量x满足约束而且f(x)在可行域有下界，二次规划问题就有一个全局最小值x。

### duality
* Lagrange
  * 考虑标准形式的优化问题:
    $$
    \begin{array}{ll}
    \operatorname{minimize} & f_0(x) \\
    \text { subject to } & f_i(x) \leqslant 0, \quad i=1, \cdots, m \\
    & h_i(x)=0, \quad i=1, \cdots, p,
    \end{array}
    $$
  * 定义问题 (5.1) 的 Lagrange 函数 $L: \mathbf{R}^n \times \mathbf{R}^m \times \mathbf{R}^p \rightarrow \mathbf{R}$ 为
    $$
    L(x, \lambda, \nu)=f_0(x)+\sum_{i=1}^m \lambda_i f_i(x)+\sum_{i=1}^p \nu_i h_i(x),
    $$
    其中定义域为 $\operatorname{dom} L=\mathcal{D} \times \mathbf{R}^m \times \mathbf{R}^p$ 。 $\lambda_i$ 称为第 $i$ 个不等式约束 $f_i(x) \leqslant 0$ 对应 的Lagrange 乘子; 类似地, $\nu_i$ 称为第 $i$ 个等式约束 $h_i(x)=0$ 对应的 Lagrange 乘子。 向量 $\lambda$ 和 $\nu$ 称为对偶变量或者是问题 (5.1) 的Lagrange 乘子向量。
  * 要求 $\lambda_i \geq 0$，$\nu_i \geq 0$（原问题是minimization问题，lagrange函数要加个负的，再求个最小，变成对偶问题，这样对偶问题永远小于原问题）
* Lagrange对偶函数 (或对偶函数)
  * 定义： 对偶函数$g: \mathbf{R}^m \times \mathbf{R}^p \rightarrow \mathbf{R}$ 为 Lagrange 函数关于 $x$ 取得的最小值。
  * 即对 $\lambda \in \mathbf{R}^m, \nu \in \mathbf{R}^p$, 有
    $$
    g(\lambda, \nu)=\inf _{x \in \mathcal{D}} L(x, \lambda, \nu)=\inf _{x \in \mathcal{D}}\left(f_0(x)+\sum_{i=1}^m \lambda_i f_i(x)+\sum_{i=1}^p \nu_i h_i(x)\right) .
    $$

## algorithm
* gradient descend 方法:

    ---
    given a starting point $x \in \operatorname{dom} f$.

    repeat
    1. Determine a descent direction $\Delta x$.
    2. Line search. Choose a step size $t>0$.
    3. Update. $x:=x+t \Delta x$.
    until stopping criterion is satisfied.
    ---

    * descent direction 一般选梯度方向
    * step size 一般用回溯法（backtracking）：
        由于一阶近似，总有一个足够小的t，在改变t*负梯度方向后，改变后的x获得的f(x)会小于原来的。那么回溯法要做的，就是把t从大往小减，知道获得一个满意的t。
        
        ---

        **given** a descent direction $\Delta x$ for $f$ at $x \in \operatorname{dom} f, \alpha \in(0,0.5), \beta \in(0,1)$. 
        
        t:=1
        
        while $f(x+t \Delta x)>f(x)+\alpha t \nabla f(x)^T \Delta x, \quad t:=\beta t$.

        ---
    * 牛顿下降法：t大小设为$-\nabla^2 f(x)$，可以很快收敛（可以从二阶近似角度进行解释，但是不一定是最快收敛的？）