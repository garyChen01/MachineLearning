import numpy as np
from sklearn.preprocessing import LabelBinarizer


class MultinomialNB(object):
    """
    Parameters
    ----
    alpha : float, optional (default=1.0)
    表示平滑系数，默认为1，即拉普拉斯平滑

    class_prior : array-like, size (n_classes,) ,optional (default=None)
    类别的先验概率。如果为None，则根据数据学习类别先验。

    """

    def __init__(self, alpha=1.0, class_prior=None):
        self.alpha = alpha
        self.class_prior = class_prior

    def fit(self, X, y, sample_weight=None):
        """
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        训练样本，n_samples表示样本数，n_features表示特征数

        y: array_like, shape = [n_samples]
        训练样本X的标签（类别）

        sample_weight: array-like, shape = [n_samples], (default=None)
        样本的权重

        Returns
        -------
        self : object
        返回自身
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if np.any(X < 0):
            raise ValueError("Input X must be non-negative")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise ValueError("Cannot support multilabels")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Find y shape is %r but X shape is %r" \
                             % ([int(l) for l in y.shape], [int(l) for l in X.shape]))
        _, n_features = X.shape
        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:  # 训练集出现的类别只有一种
            Y = np.concatenate((Y, 1 - Y), axis=1)
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            if sample_weight.reshape(-1).shape[0] != X.shape[0]:
                raise ValueError("sample_weight shape is %r, but X shape is %r" \
                                 % ([int(l) for l in sample_weight.shape], [int(l) for l in X.shape]))
            sample_weight = np.atleast_2d(sample_weight)
            # 将样本权重体现在对应类别的频数统计中
            Y *= sample_weight.T
        self.class_count_ = Y.sum(axis=0)
        self.feature_count_ = np.dot(Y.T, X)

        smoothed_fc = self.feature_count_ + self.alpha
        # 相当于各类中，各单词出现的频数除以该类的总单词数，即各单词的频率，然后转换为log频率
        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_fc.sum(axis=1).reshape(-1, 1)))
        if self.class_prior is not None:
            self.class_prior = np.log(self.class_prior)
        else:
            self.class_prior = (np.log(self.class_count_) -
                                np.log(self.class_count_.sum()))
        return self

    def _joint_log_likelihood(self, X):
        return np.dot(X, self.feature_log_prob_.T) + self.class_prior

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
