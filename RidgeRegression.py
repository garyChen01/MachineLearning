import numpy as np
from numpy import mat
from scipy import sparse, linalg


class RidgeRegreesion_SGD():
    """
    本算法使用随机梯度下降法来求解模型
    """

    def __init__(self, alpha=1, max_iter=1000, lr=0.0000001):
        self.alpha = alpha
        self.max_iter = max_iter
        self.lr = lr
        print(self.alpha, self.max_iter, self.lr)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None and np.atleast_1d(sample_weight) > 1:
            raise ValueError("Sample weights must be 1D array or scalar")
        n_samples, n_features = X.shape
        X = np.c_[X, np.ones(n_samples)]
        if sample_weight is not None:
            X, y = self.rescale_data(X, y, sample_weight)
        self.coef_ = np.zeros(n_features + 1)  # [W1,...,Wn,b]
        for ite in range(self.max_iter):
            loss = 0
            for i in range(n_samples):
                pred = np.dot(X[i], self.coef_)
                error = y[i] - pred
                loss += error ** 2
                self.coef_ -= self.lr * (-2 * error * X[i] + 2 * self.coef_)
            if np.isnan(self.coef_).any():
                raise ValueError("Learning rate is too large so that some weight equal NaN")
        return self

    def predict(self, X):
        return np.dot(np.c_[X, np.ones(X.shape[0])], self.coef_)

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


class RidgeRegreesion_OLS():
    def __init__(self):
        pass

    def fit(self, X, y, sample_weight=None):

        if sample_weight is not None and np.atleast_1d(sample_weight) > 1:
            raise ValueError("Sample weights must be 1D array or scalar")
        if sample_weight is not None:
            X, y = self.rescale_data(X, y, sample_weight)

        # 手动求解最小二乘法问题
        xMat = mat(X)
        yMat = mat(y).T
        xTx = xMat.T * xMat
        xTx += np.eye(xTx.shape[0])
        if linalg.det(xTx) == 0.0:
            raise ValueError("Can't compute least-square solution, Please to add normalization")
        self.coef_ = xTx.I * (xMat.T * yMat)
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : 样本

        Returns
        -------
        y: 预测值
        """
        return X * self.coef_

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
