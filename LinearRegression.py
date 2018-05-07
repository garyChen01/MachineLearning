import numpy as np
from scipy import sparse,linalg

"""
对线性回归模型进行了两种实现。
一是使用最小二乘法（OLS）二是使用随机梯度下降（SGD）
"""

class LinearRegreesion_OLS():
    """
    使用普通最小二乘法来求解线性回归模型
    """

    def __init__(self):
        pass

    def fit(self, X, y, sample_weight=None):
        """
        拟合数据集，生成一个线性回归模型
        Paramters
        ---------
        X: 特征数据
        y: 输出值
        sample_weight: 样本权重

        Returns
        -------
        self: 返回自身的一个实例
        """
        if sample_weight is not None and np.atleast_1d(sample_weight) > 1:
            raise ValueError("Sample weights must be 1D array or scalar")
        if sample_weight is not None:
            X, y = self.rescale_data(X, y, sample_weight)
        """
        # 使用scipy的函数求解最小二乘法问题
        self.coef_, self._residues, self.rank_, self.singular_ = \
            linalg.lstsq(X, y)
        """
        # 手动求解最小二乘法问题
        xTx = X.T * X
        if linalg.det(xTx) == 0.0:
            raise ValueError("Can't compute least-square solution, Please to add normalization")
        self.coef_ = xTx.I * (X.T * y)
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


class LinearRegression_SGD():
    def __init__(self, max_iter=100 , lr=0.00001):
        self.max_iter  = max_iter
        self.lr = lr

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None and np.atleast_1d(sample_weight) > 1:
            raise ValueError("Sample weights must be 1D array or scalar")
        n_samples, n_features = X.shape
        X = np.c_[X, np.ones(n_samples)]
        if sample_weight is not None:
            X, y = self.rescale_data(X, y, sample_weight)
        self.coef_ = np.zeros(n_features + 1)
        # self.coef_ = np.zeros(n_features)
        # 迭代次数
        for ite in range(self.max_iter):
            loss = 0
            # 遍历全训练集
            for i in range(n_samples):
                pred = np.dot(X[i],self.coef_)
                error = y[i] - pred
                loss += error ** 2
                self.coef_ += self.lr * 2 * error * X[i]
            if np.isnan(self.coef_).any():
                raise ValueError("Learning rate is too large so that some weight equal NaN")
        return self

    def predict(self, X):
        return np.dot(np.c_[X, np.ones(X.shape[0])], self.coef_)
        # return np.dot(X,self.coef_)

    def rescale_data(self, X, y, sample_weight):
        """
        对带样本权重的数据进行处理，使之处理后的数据与无样本权重的数据一样。
        """
        n_samples = X.shape[0]
        sample_weight = sample_weight*np.ones(n_samples) # 处理当sample_weight为常数的情况
        # 损失函数使用的是平方和误差，所以X和y的系数为样本权重的平方根
        sample_weight = np.sqrt(sample_weight)
        # sw_matrix为对角矩阵，对角线上的值为对应的样本权重
        sw_matrix = sparse.diamatrix((sample_weight,0),shape=(n_samples,n_samples))
        X = sw_matrix*X
        y = sw_matrix*y
        return X, y

