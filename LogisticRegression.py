import numpy as np
from sklearn.preprocessing import LabelEncoder

"""
本算法实现了二分类与多分类的逻辑回归算法
对于二分类问题，只能使用ovr
而对于多分类问题，可以使用multinomial或ovr。
代码参考Sklearn源码
"""


class LogisticRegression():
    """
    multi_class : str, {'ovr', 'multinomial'}, default: 'ovr'
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution.
    max_iter: int
        Maximum number of iterations.
    lr:float
        learning rate in Gradient Descent method
    """

    def __init__(self, max_iter=100, lr=0.001, multi_class='ovr'):
        self.max_iter = max_iter
        self.lr = lr
        self.lb = LabelEncoder()
        self.multi_class = multi_class

    def fit(self, X, y):
        y = self.lb.fit_transform(y)
        X = np.c_[X, np.ones(X.shape[0])]
        if self.multi_class not in ['multinomial', 'ovr']:
            raise ValueError("multi_class should be either multinomial or "
                             "ovr, got %s" % multi_class)
        n_classes = np.unique(y).size
        if n_classes == 2:
            if self.multi_class == 'ovr':
                n_classes = 1
            else:
                raise ValueError("multi_class should be ovr if binary classify problem")
        self.coef_ = np.zeros((n_classes, X.shape[1]))
        for ite in range(self.max_iter):
            if self.multi_class == 'multinomial':
                y_prob = self.softmax(np.dot(X, self.coef_.T))
            else:
                y_prob = self.sigmoid(np.dot(X, self.coef_.T))
            for j in range(n_classes):
                y_j = [1 if y_k == j else 0 for y_k in y]
                grad = 0
                for i in range(X.shape[0]):
                    grad += (y_j[i] - y_prob[i][j]) * X[i]
                self.coef_[j] += self.lr * grad
        return self

    def predict(self, X):
        prob = self.predict_proba(X)
        pos = np.argmax(prob, axis=1)
        y = self.lb.inverse_transform(pos)
        return y

    def predict_proba(self, X):
        X = np.c_[X, np.ones(X.shape[0])]
        if self.multi_class == 'ovr':
            prob = self.sigmoid(np.dot(X, self.coef_.T))
            if prob.shape[1] == 1:
                prob = np.c_[prob, 1 - prob]
            else:
                # Ovr normalization
                prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
        else:
            prob = self.softmax(np.dot(X, self.coef_.T))
        return prob

    def softmax(self, X):
        max_prob = np.max(X, axis=1).reshape((-1, 1))
        X -= max_prob

        X = np.exp(X)
        sum_prob = np.sum(X, axis=1).reshape((-1, 1))
        X /= sum_prob
        return X

    def sigmoid(self, X):
        X = 1 / (1 + np.exp(-X))
        return X
