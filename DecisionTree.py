import sys

import numpy as np


class DecisionTreeClassifier(object):
    def __init__(self, split='gini'):
        self.split = split

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Find y shape is %r but X shape is %r" \
                             % ([int(l) for l in y.shape], [int(l) for l in X.shape]))
        self.tree = self._buildTree(X, y, np.arange(X.shape[0]))

    def _buildTree(self, X, y, idx):
        _, n_features = X.shape
        base_val = self.criterion(y, np.arange(y.shape[0])[np.newaxis, :])
        best_gain, best_split_idx, best_split_val = 0, -1, 0
        # 遍历特征
        for feat_idx in range(n_features):
            col = X[:, feat_idx]
            col = np.sort(col)
            col = np.unique(col)
            # 遍历特征值
            for val_idx in range(1, col.shape[0]):
                # 分割值为排序好的属性的相邻两点的中位值
                split_val = (col[val_idx] + col[val_idx - 1]) / 2
                sub_idx, cri_val = self.dataSplit(X, y, feat_idx, split_val)
                if base_val - cri_val > best_gain:
                    best_gain = base_val - cri_val
                    best_split_idx = feat_idx
                    best_split_val = split_val
                    best_sub_idx = sub_idx
        root = DecisionTreeClassifier.treeNode(best_split_idx, best_split_val)
        if best_split_idx != -1:
            root.leftNode = self._buildTree(X[best_sub_idx[0]], y[best_sub_idx[0]], idx[best_sub_idx[0]])
            root.rightNode = self._buildTree(X[best_sub_idx[1]], y[best_sub_idx[1]], idx[best_sub_idx[1]])
        # 无法分割，则输出类别
        else:
            root.leftNode = self.vote(y)
        return root

    class treeNode(object):
        def __init__(self, split_idx, split_val, left=None, right=None):
            self.split_idx = split_idx
            self.split_val = split_val
            self.leftNode = left
            self.rightNode = right

    def vote(self, y):
        class_num = {}
        for key in y:
            if key not in class_num.keys():
                class_num[key] = 0
                class_num[key] += 1
        class_num = sorted(class_num.items(), key=lambda x: x[1])
        return class_num[0][0]

    # 对数据进行划分，并计算对应的criterion值
    def dataSplit(self, X, y, split_idx, split_val):
        # subTree[0]保存小于split_val的索引，subTree[1]保存大于split_val的索引
        subTree = [[], []]
        for idx, val in enumerate(X[:, split_idx]):
            if val >= split_val:
                subTree[1].append(idx)
            else:
                subTree[0].append(idx)
        return subTree, self.criterion(y, subTree)

    def criterion(self, y, subTree):
        length = y.shape[0]
        val = 0
        # 基尼指数
        if self.split == 'gini':
            for sub in subTree:
                sub_length = len(sub)
                binCount = np.bincount(y[sub])
                binCount = np.apply_along_axis(lambda x: (x / sub_length) ** 2, 0, binCount).sum()
                val += sub_length / length * (1 - binCount)
        # 信息增益
        elif self.split == 'entropy':
            for sub in subTree:
                sub_length = len(sub)
                binCount = np.bincount(y[sub])
                binCount = np.apply_along_axis(lambda x: -(x / sub_length) * \
                                                         np.log(x / sub_length), 0, binCount)
                binCount.sum()
                val += sub_length / length * (1 - binCount)
        else:
            raise ValueError('Just support gini and entropy')
        return val

    def predict(self, X):
        X = np.atleast_2d(X)
        predict = []
        for sample in X:
            root = self.tree
            while (root.rightNode):
                if root.split_val <= sample[root.split_idx]:
                    root = root.rightNode
                else:
                    root = root.leftNode
            predict.append(root.leftNode)
        return predict


class DecisionTreeRegressor(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Find y shape is %r but X shape is %r" \
                             % ([int(l) for l in y.shape], [int(l) for l in X.shape]))
        self.tree = self._buildTree(X, y)

    def _buildTree(self, X, y):
        _, n_features = X.shape
        base_val = sys.maxsize
        best_gain, best_split_idx, best_split_val = 0, -1, 0
        # 遍历特征
        for feat_idx in range(n_features):
            col = X[:, feat_idx]
            col = np.sort(col)
            col = np.unique(col)
            # 遍历特征值
            for val_idx in range(1, col.shape[0]):
                # 分割值为排序好的属性的相邻两点的中位值
                split_val = (col[val_idx] + col[val_idx - 1]) / 2
                sub_idx, cri_val = self.dataSplit(X, y, feat_idx, split_val)
                if base_val - cri_val > best_gain:
                    best_gain = base_val - cri_val
                    best_split_idx = feat_idx
                    best_split_val = split_val
                    best_sub_idx = sub_idx
        root = DecisionTreeClassifier.treeNode(best_split_idx, best_split_val)
        if best_split_idx != -1:
            root.leftNode = self._buildTree(X[best_sub_idx[0]], y[best_sub_idx[0]])
            root.rightNode = self._buildTree(X[best_sub_idx[1]], y[best_sub_idx[1]])
        # 无法分割，则输出类别
        else:
            root.leftNode = self.vote(y)
        return root

    def dataSplit(self, X, y, split_idx, split_val):
        # subTree[0]保存小于split_val的索引，subTree[1]保存大于split_val的索引
        subTree = [[], []]
        for idx, val in enumerate(X[:, split_idx]):
            if val >= split_val:
                subTree[1].append(idx)
            else:
                subTree[0].append(idx)
        return subTree, self.criterion(y, subTree)

    def criterion(self, y, subTree):
        val = 0
        # MSE
        for sub in subTree:
            val += np.sum(((y[sub] - np.average(y[sub])) ** 2))
        return val

    class treeNode(object):
        def __init__(self, split_idx, split_val, left=None, right=None):
            self.split_idx = split_idx
            self.split_val = split_val
            self.leftNode = left
            self.rightNode = right

    def vote(self, y):
        return np.average(y)

    def predict(self, X):
        X = np.atleast_2d(X)
        predict = []
        for sample in X:
            root = self.tree
            while (root.rightNode):
                if root.split_val <= sample[root.split_idx]:
                    root = root.rightNode
                else:
                    root = root.leftNode
            predict.append(root.leftNode)
        return predict
