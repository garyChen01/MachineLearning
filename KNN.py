import sys
from heapq import heappop, heappush

import numpy as np


class KdTree(object):
    def __init__(self, data, leafsize):
        self.data = np.asarray(data)
        self.leafsize = leafsize
        self.n, self.m = self.data.shape
        self.maxes = np.amax(self.data, axis=0)
        self.mins = np.amin(self.data, axis=0)
        self.root = self.__build(np.arange(self.n), self.maxes, self.mins)

    class node(object):
        # 使用heappush是有用
        if sys.version_info[0] >= 3:
            def __lt__(self, other):
                return id(self) < id(other)

            def __gt__(self, other):
                return id(self) > id(other)

            def __le__(self, other):
                return id(self) <= id(other)

            def __ge__(self, other):
                return id(self) >= id(other)

            def __eq__(self, other):
                return id(self) == id(other)

    class leafnode(node):
        def __init__(self, idx):
            self.idx = idx

    class innernode(node):
        def __init__(self, split, split_dim, less, greater):
            self.split = split
            self.split_dim = split_dim
            self.less = less
            self.greater = greater

    def minkowski_distance_p(self, y, x, p):
        x = np.asarray(x)
        y = np.asarray(y)
        if p == np.inf:
            return np.amax(np.abs(y - x), axis=-1)
        else:
            return np.sum(np.abs(y - x) ** p, axis=-1)

    def __build(self, idx, maxes, mins):
        # 如果当前数据集的大小小于指定的最小分裂数leafsize，则将该数据集作为叶节点返回。
        data = self.data[idx]
        if len(data) <= self.leafsize:
            return KdTree.leafnode(idx)
        d = np.argmax(maxes - mins)
        if maxes[d] == mins[d]:
            # 各样本点完全相等
            return KdTree.leafnode(idx)
        data = data[:, d]
        split = (maxes[d] + mins[d]) / 2
        less_idx = np.nonzero(data <= split)[0]
        greater_idx = np.nonzero(data > split)[0]
        # 注意mins与maxes是定义了data的区间，并不一定data有样本点正好处于区间边缘
        # 所以可能出现数据点在split的一边的情况
        if len(less_idx) == 0:
            split = np.amin(data)
            less_idx = np.nonzero(data <= split)[0]
            greater_idx = np.nonzero(data > split)[0]
        if len(greater_idx) == 0:
            split = np.amax(data)
            less_idx = np.nonzero(data < split)[0]
            greater_idx = np.nonzero(data >= split)[0]
        if len(less_idx) == 0:
            # 此时，可以判断实际上，data完全相等
            split = data[0]
            less_idx = np.range(len(data) - 1)
            greater_idx = [len(data) - 1]
        lessmaxes = np.copy(maxes)
        lessmaxes[d] = split
        greatermins = np.copy(mins)
        greatermins[d] = split
        return KdTree.innernode(split, d,
                                self.__build(idx[less_idx], mins, lessmaxes),
                                self.__build(idx[greater_idx], greatermins, maxes))

    def __query(self, x, k=1, p=2, distance_upper_bound=np.inf):
        side_distance = np.maximum(0, np.maximum(x - self.maxes, self.mins - x))
        if p == np.inf:
            min_distance = np.amax(side_distance)
        elif p < 1:
            raise ValueError("p must more than 1")
        else:
            side_distance **= p
            min_distance = np.sum(side_distance)
        q = []
        heappush(q, (min_distance, tuple(side_distance), self.root))
        neighbors = []
        while q:
            min_distance, side_distance, node = heappop(q)
            if isinstance(node, KdTree.leafnode):
                data = self.data[node.idx]
                ds = self.minkowski_distance_p(data, x[np.newaxis, :], p)
                for i in range(len(ds)):
                    # 如果neighbors大小为k，则distance_upper_bound为当前neighbors的最小值
                    # 否则distance_upper_bound为np.inf,使得可以不断有数据点加入neighbors直到neighbors大小为k
                    if ds[i] < distance_upper_bound:
                        if len(neighbors) == k:
                            heappop(neighbors)
                        heappush(neighbors, (-ds[i], node.idx[i]))
                        if len(neighbors) == k:
                            # neighbors[0][0]为当前neighbors中的最大距离，其以最小堆的形式存储数据
                            distance_upper_bound = -neighbors[0][0]
            else:
                # 当node为内部节点时，需要计算数据点x与node的边界的距离
                if min_distance > distance_upper_bound:
                    break
                if x[node.split_dim] < node.split:
                    near, far = node.less, node.greater
                else:
                    near, far = node.greater, node.less
                # 更近的子节点与边界的距离是要等于min_distance，所以可以直接将其压入堆中
                heappush(q, (min_distance, side_distance, near))
                # 更远的子节点需要判断距离是否小于distance_upper_bound
                sd = list(side_distance)
                if p == np.inf:
                    # 当p=np.inf时，一个向量x的距离为x中的最大值。
                    min_distance = max(min_distance, abs(node.split - x[node.split_dim]))
                else:
                    # side_distance需要更新，即将split_dim维度的距离变更
                    sd[node.split_dim] = np.abs(x[node.split_dim] - node.split) ** p
                    min_distance = min_distance - side_distance[node.split_dim] + sd[node.split_dim]
                if min_distance < distance_upper_bound:
                    heappush(q, (min_distance, tuple(sd), far))
        if p == np.inf:
            return sorted([(-d, i) for (d, i) in neighbors])
        else:
            return sorted([((-d) ** (1 / p), i) for (d, i) in neighbors])

    def query(self, x, k=1, p=2, distance_upper_bound=np.inf):
        x = np.asarray(x)
        if x.shape[-1] != self.m:
            raise ValueError("x must consist of vectors of length %d but has shape %s" % (self.m, np.shape(x)))
        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")
        retshape = np.shape(x)[:-1]
        if retshape != ():  # x为多个数据点组成的数据集
            if k == 1:
                dd = np.empty(retshape, dtype=float)
                dd.fill(np.inf)
                ii = np.empty(retshape, dtype=int)
                ii.fill(self.n)
            elif k > 1:
                # retshape+(k,) == (retshape,k)
                dd = np.empty(retshape + (k,), dtype=float)
                dd.fill(np.inf)
                ii = np.empty(retshape + (k,), dtype=int)
                ii.fill(self.n)
            else:
                raise ValueError(
                    "Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one,")
            for c in np.ndindex(retshape):
                hits = self.__query(x[c], k=k, p=p, distance_upper_bound=distance_upper_bound)
                if k == 1:
                    if len(hits) > 0:
                        dd[c], ii[c] = hits[0]
                else:
                    for j in range(len(hits)):
                        dd[c + (j,)], ii[c + (j,)] = hits[j]
            return dd, ii
        else:
            hits = self.__query(x, k=k, p=p, distance_upper_bound=distance_upper_bound)
            if k == 1:
                if len(hits) > 0:
                    return hits[0]
                else:
                    return np.inf, self.n
            elif k > 1:
                dd = np.empty(k, dtype=float)
                dd.fill(np.inf)
                ii = np.empty(k, dtype=int)
                ii.fill(self.n)
                for j in range(len(hits)):
                    dd[j], ii[j] = hits[j]
                return dd, ii
            else:
                raise ValueError(
                    "Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one")


class KNeighborsClassifier(object):
    def __init__(self, n_neighbors=5, leaf_size=30, p=2):
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.p = p
        self.tree = None

    def fit(self, X, y):
        self.tree = KdTree(X, self.leaf_size)
        self._class = np.asarray(y)

    def kneighbors(self, query_points, n_neighbors=None):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if self.tree is None:
            raise ValueError("Please fit data before search k neighbors")
        return self.tree.query(query_points, k=n_neighbors, p=self.p)

    def predict(self, X):
        m = len(X)
        y_pred = np.empty(m, dtype=self._class.dtype)
        dd, ii = self.kneighbors(X)
        for j in range(m):
            neighbors_class = np.unique(self._class[ii[j]], return_counts=True)
            y_pred[j] = np.amax(neighbors_class[0][np.argmax(neighbors_class[1])])
        return y_pred


class KNeighborsRegressor(object):
    def __init__(self, n_neighbors=5, leaf_size=30, p=2):
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.p = p
        self.tree = None

    def fit(self, X, y):
        self.tree = KdTree(X, self.leaf_size)
        self._y = y

    def kneighbors(self, query_points, n_neighbors=None):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if self.tree is None:
            raise ValueError("Please fit data before search k neighbors")
        return self.tree.query(query_points, k=n_neighbors, p=self.p)

    def predict(self, X):
        m = len(X)
        y_pred = np.empty(m, dtype=self._y.dtype)
        dd, ii = self.kneighbors(X)
        for j in range(m):
            y_pred[j] = np.mean(self._y[ii[j]])
        return y_pred
