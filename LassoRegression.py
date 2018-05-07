import numpy as np
from scipy import sparse


class LassoRegreesion_CD():
    """
    本算法使用坐标下降法来求解模型
    """

    def __init__(self, alpha=1, max_iter=100):
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None and np.atleast_1d(sample_weight) > 1:
            raise ValueError("Sample weights must be 1D array or scalar")
        n_samples, n_features = X.shape
        # X = np.c_[X, np.ones(n_samples)]
        if sample_weight is not None:
            X, y = self.rescale_data(X, y, sample_weight)
        # self.coef_ = np.zeros(n_features+1) # [W1,...,Wn,b]
        self.coef_ = np.zeros(n_features)
        for ite in range(self.max_iter):
            for i in range(n_features):
                zi = np.sum(X[:, i] ** 2)
                pi = np.dot((y - np.dot(X, self.coef_)), X[:, i]) + zi * self.coef_[i]
                if pi < -self.alpha / 2:
                    self.coef_[i] = (pi + self.alpha / 2) / zi
                elif pi > self.alpha / 2:
                    self.coef_[i] = (pi - self.alpha / 2) / zi
                else:
                    self.coef_[i] = 0
            if np.isnan(self.coef_).any():
                raise ValueError("Learning rate is too large so that some weight equal NaN")
        return self

    def predict(self, X):
        # return np.dot(np.c_[X, np.ones(X.shape[0])], self.coef_)
        return np.dot(X, self.coef_)

    def rescale_data(self, X, y, sample_weight):
        """
        对带样本权重的数据进行处理，使之处理后的数据与无样本权重的数据一样。
        """
        n_samples = X.shape[0]
        sample_weight = sample_weight * np.ones(n_samples)  # 处理当sample_weight为常数的情况
        # 损失函数使用的是平方和误差，所以X和y的系数为样本权重的平方根
        sample_weight = np.sqrt(sample_weight)
        # sw_matrix为对角矩阵，对角线上的值为对应的样本权重
        sw_matrix = sparse.diamatrix((sample_weight, 0), shape=(n_samples, n_samples))
        X = sw_matrix * X
        y = sw_matrix * y
        return X, y
