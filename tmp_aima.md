# aima
## search
### graph search
* 图搜索和树搜索区别
  * 图搜索需要额外判断该结点是否到达过

### search strategy
* properties:
  * complete：如果存在解，则一定可以找到解
  * optimal: 如果找到了解，则一定是最优解
  * Time complexity
  * Space complexity
* preliminary:
  * (branch) b: 树展开的最大节点数
  * (depth) d: 答案在的节点的那一层树的深度
  * (max depth) m: 树的最大深度
 
#### 无先验知识
##### breath-first search
* complete: yes 
* optimal: yes
* Time: O( $b^{d+1}$ )
* Space: O( $b^d$ )

##### depth-first search
* complete: 
  * Yes: in finite spaces with repeated states avoid
  * No: in infinite-depth spaces, spaces with loops
* optimal: no
* Time: O( $b^{m}$ ) 
* Space: O( $bm$ )
  * ps: 为什么是 bm呢？请看：
    ```c++
    while (!unvisited.empty()) {
        current = (unvisited.top()); //当前访问的
        unvisited.pop();
        if (current->right != NULL)
            unvisited.push(current->right );
        if (current->left != NULL)
            unvisited.push(current->left);
        cout << current->self << endl;
    }
    ```
    哈哈，先把一个节点连接的所有都放进去，再一个个慢慢pop

##### uniform-cost search（Dijsktra）
* if:
  * step cost $\geq \epsilon$
  * $C^*$ denote the optimal solution 
* complete: Yes if step cost $\geq \epsilon$ ($\geq 0$ ?)
* optimal: Yes
* Time: O( $b^{\lceil C^*/ \epsilon \rceil}$ )
* Space: O( $b^{\lceil C^*/ \epsilon \rceil}$ )

##### depth-limited search(dfs 但是限制深度)
限制某个深度，就返回。
* complete:  if 最优解在限制的深度里有，就是yes
* optimal: no
* Time: O( $b^d$ )
* Space: O( $bd$ )


##### iterative deepening search
迭代地增加限制的深度
* complete:  yes
* optimal: yes
* Time: O( $b^d$ )
* Space: O( $bd$ )

#### 有先验知识
##### A*
* 解决问题：
  * 某图中源点与目标节点之间的最短路径
* idea: avoid expanding paths that are already expensive
* 评价指标 f(n) = g(n) + h(n)
  * g(n) 表示到该点的真实开销，会在走到该点的父节点（意会）的时候被更新
  * h(n) 启发式函数，需要满足
    1. The heuristic is admissible, as it will never overestimate the cost.
    2. h(object) = 0
    * 证明：所有在最优路径上的点都会比目标点先展开。这样下去迟早会展开到目标节点 
    * 常见函数有：
      * 比如 8-puzzle,可以用的启发函数：1. 没有正确排序的格子的个数 2. 总的曼哈顿距离
* properties:
  * optimal: yes!
  * complete: yes!

##### Adversarial search
* 解决问题：
* 算法：(伪代码，意会)
  ```python
    def minmax(s)->a:
        ans = None, value = -INF
        for a in actions:
            next_s = env.step(s,a)
            if min_value(next_s) > value:
                ans = a
                value = min_value(next_s)

    def min_value(s)->value:
        if terminal(s):
            return utility(s)
        value = INF
        for next_s in succerssors(s):
            value = min(value, max_value(next_s))

    def max_value(s)->value:
        if terminal(s):
            return utility(s)
        value = -INF
        for next_s in succerssors(s):
            value = max(value, min_value(next_s))
  ```
* property:
  * complete: yes, if tree finite
  * optimal: 只在对手也按minmax策略走的时候optimal
  * time: O( $b^m$ )
  * space: O( $bm$ )

### Bandits and MCTS
搜索是不可能完全搜出来的
#### Bandits

#### MCTS
* process:
  * selection
  * expansion
  * evaluation
  * backup

### 普适的解空间搜索
#### 比较简单的方法
* 爬山法（exploitation）
  * 缺点：可能陷入局部最优
* 随机搜索（exploration）：
  * 优点：当尝试次数趋近于正无穷，肯定能找到最优
  * 缺点：...嘎嘎

#### genetic algorithm
introAI: page 123

#### constraint satisfaction problems(CSPs)
* backtracking search
  * 回溯搜索，遇到不行的返回上一步
  * make it more informed:
    * ppt 132, 比较好记的：failure能不能早点检测到? 顺序应该如何被尝试?
  * 树形CSPs/近似树形CSPs没看懂，之后看

## knowledge(知识期)
### entailment（蕴含）
* $KB \models \alpha$ means for all models(带入semantic)， if KB is true, then a is true
  * i.e. $M(KB) \subseteq M(\alpha)$

### soundness/completeness
见ppt 170

### 证明 $KB\models \alpha$
* 法一：
  * validity: a sentence is valid if it is true in all models
  * $KB \models \alpha$ iff $(KB \Rightarrow \alpha)$ is valid
* 法二：
  * unsatisfiable: a sentence if unsatisfiable if it is true in no models
  * $KB \models \alpha$ iff $(KB \wedge \neg \alpha)$ is unsatisfiable

### chaining
* ppt 182
*  forward chaining
  * fire any rule whose premises are satisfied in the KB, add its conclusion to the KB, until query is found.
  * data-driven, automatic, unconscious processing
* backward chaining
  * work backwards from the query q: check if q is known already, or prove by BC all premises of some rule concluding q
  * goal-driven, appropriate for problem-solving
  * complexity can be much less than linear in size of KB

### 常考：一阶谓词逻辑写法
ppt 222

### unification/prolog system
学都没学过，什么东西！ppt236

### conjuction normal form (CNF)
$\wedge_{i=1}^{m}(\vee_{k=1}^{n} P)$
#### boolean satisfiability problem (SAT)
* literals: $x_1, x_2,...$
* clauses: $(x_1 \vee x_2 \vee x_3)$
* find an assignment to literals so that the conjuction of the clauses is true, or prove unsatisfiable
* complexity:
  * 2SAT(every clause has at most 2 literals): P-solvable
  * 3SAT: NP-hard

## Statistic
### Bayesian  networks
#### exact inference
* inference by variable elimination:
  * can reduce 3SAT to exact inference, thus NP-contained

#### approximate inference
* inference by stochastic simulation
  * basic idea:
    1. draw N samples from a sampling distribution S
    2. compute an approximate posterior probability P
    3. show thid converges to the true probabiliry P
  * methods:
    * sampling from an empty network
    * rejection sampling: reject samples disagreeing with evidence
      * $\hat{P(X|e)}$ estimated from samples agreeing with e
    * likelihood weighting: use evidence to weight samples
    * Markov chain Monte Carlo: sample from a stochastic process whose stationary distribution is the true posterior 