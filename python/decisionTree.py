import numpy as np
from collections import Counter


class Node:
    def __init__(self, a=None, threshold=None, left=None, right=None, value=None) -> None:
        """
        """
        self.a = a
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # 当叶节点时，判断是什么类型

    def is_leaf(self)->bool:
        return self.value is not None

class DecisionTree:
    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        pass
    
    def _gen_tree(self, X, y, attrs:list)->Node:
        if not attrs:
            value = self._most_commen(y)
            return Node(value=value)
        for attr in attrs:
            pass
            


    def _most_commen(self, y):           
        c = Counter(y)
        mc = c.most_common(1)[0][0]
        return mc



if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy:", acc)