import random
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier


class Adaboost:
    def __init__(self, cls=DecisionTreeClassifier) -> None:
        self.cls = cls
        self.models = []
        self.a_values = []

    def fit(self, x_train, y_train, T = 5):
        self.m = len(x_train)
        w = np.ones((self.m), dtype = float) / self.m
        for t in T:
            model = self.cls()
            model.fit(x_train, y_train, sample_weight = w)
            pred_x = model.predict(x_train)
            fault = np.asarray(pred_x != y_train, dtype = float)
            e = sum(self.w * fault)
            a = 0.5 * np.exp((1-e) / e)
            update = fault * 2 - 1  # 错误的值为1，正确的值为-1
            w = w * np.exp(a * update) 
            w = w / np.sum(w) # scale 到[0,1]之间
            self.models.append(model)
            self.a_values.append(a)
        self.a_values = np.array(self.a_values)

    def predict(self, X):
        """ predict the value
        """
        preds = np.array([model.predict(X) for model in self.models])
        preds_value = np.matmul(preds.T, self.a_values)
        pred = np.asarray(preds_value >= sum(self.a_values)/2, type=int)
        return pred
    
def main():
    # TODO(complete this when needed): add dataset,training test
    pass




    


        
