# mlAlgorithm
## c++

## python
### pca
* 去中心化！！！一天到晚老是忘！！
* argsort 默认升序！！获得的idx需要 [::-1]来转换成降序！！！
* np.linalg.eig 返回的 vector 是按列排的！！！!也就是说选择一部分需要：[:,:d]！！！！

### LDA
* np.unique 相当好用
* 注意算 sb 时的权重！！要乘以某一类的数量的
* 注意python中一维数组运算！！！ 如果x是一维，想得到xx^T 是二维需要先把x reshape！！不然会一直得到一个数字然后广播.....

### decision tree
* Counter 相当好用！可以像dict一样操作
  ```python
  form collections import Counter
  c = Counter(iterable)
  c.most_commen(n)  # return a list of n most commen objects
  ```

* gen_tree返回node节点是好文明！