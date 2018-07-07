import random

import numpy as np


class SVMClassifier():
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0, tol=1e-3, max_iter=-1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

    def calcKernelMatrix(self, X, Z):
        n = X.shape[0]
        m = Z.shape[0]
        kernelMatrix = np.zeros((n, m))
        for i in range(m):
            kernelMatrix[:, i] = self.calcKernelValue(X, Z[i, :])
        return kernelMatrix

    def calcKernelValue(self, X, X_i):
        n_samples = X.shape[0]
        kernelVal = np.zeros((n_samples))
        if self.kernel == 'linear':
            kernelVal = np.dot(X, X_i)
        elif self.kernel == 'rbf':
            if self.gamma == 'auto':
                self.gamma = 1 / X.shape[1]
            for j in range(n_samples):
                diff = X[j, :] - X_i
                kernelVal[j] = np.exp(np.dot(diff, diff) / (-2.0 * self.gamma ** 2))
        else:
            raise NameError('Not support kernel type! You can use linear or rbf!')
        return kernelVal

    def calcEk(self, k):
        return float(np.dot(self.alpha * self.y, self.kernelMatrix[:, k]).reshape(-1) + self.bias) - self.y[k]

    def updateEk(self, k):
        Ek = self.calcEk(k)
        self.errorMatrix[k] = [1, Ek]

    def selectJ(self, i, Ei):
        maxK = -1;
        maxDeltaE = 0;
        Ej = 0
        self.errorMatrix[i] = [1, Ei]
        validEList = np.nonzero(self.errorMatrix[:, 0])[0]
        if (len(validEList)) > 1:
            for k in validEList:
                if k == i: continue
                Ek = self.calcEk(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k;
                    maxDeltaE = deltaE;
                    Ej = Ek
            if maxK == -1:
                maxK = i
                while i == maxK:
                    maxK = int(random.uniform(0, self.n_samples))
                Ej = self.calcEk(maxK)
            return maxK, Ej
        else:
            j = i
            while i == j:
                j = int(random.uniform(0, self.n_samples))
            Ej = self.calcEk(j)
            return j, Ej

    def innerLoop(self, i, y):
        Ei = self.calcEk(i)
        if ((y[i] * Ei < -self.tol) and (self.alpha[i] < self.C)) or ((y[i] * Ei > self.tol) and (self.alpha[i] > 0)):
            j, Ej = self.selectJ(i, Ei)
            alphaIold = self.alpha[i].copy()
            alphaJold = self.alpha[j].copy()
            if (self.y[i] != self.y[j]):
                L = max(0.0, self.alpha[j] - self.alpha[i])
                H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
            else:
                L = max(0.0, self.alpha[i] + self.alpha[j] - self.C)
                H = min(self.C, self.alpha[i] + self.alpha[j])
            if L == H:
                return 0
            eta = 2.0 * self.kernelMatrix[i, j] - self.kernelMatrix[i, i] - self.kernelMatrix[j, j]
            if eta >= 0:
                return 0
            self.alpha[j] -= self.y[j] * (Ei - Ej) / eta
            if self.alpha[j] > H:
                self.alpha[j] = H
            if L > self.alpha[j]:
                self.alpha[j] = L
            self.updateEk(j)
            if (abs(self.alpha[j] - alphaJold) < 1e-8):
                return 0
            self.alpha[i] += self.y[i] * self.y[j] * (alphaJold - self.alpha[j])
            self.updateEk(i)
            bi = self.bias - Ei - self.y[i] * (self.alpha[i] - alphaIold) * self.kernelMatrix[i, i] - \
                 self.y[j] * (self.alpha[j] - alphaJold) * self.kernelMatrix[i, j]
            bj = self.bias - Ej - self.y[i] * (self.alpha[i] - alphaIold) * self.kernelMatrix[i, j] - \
                 self.y[j] * (self.alpha[j] - alphaJold) * self.kernelMatrix[j, j]
            if (0 < self.alpha[i]) and (self.alpha[i] < self.C):
                self.bias = bi
            elif (0 < self.alpha[j]) and (self.alpha[j] < self.C):
                self.bias = bj
            else:
                self.bias = (bi + bj) / 2.0
            return 1
        else:
            return 0

    def fit(self, X, y):
        self.kernelMatrix = self.calcKernelMatrix(X, X)
        self.unique = np.unique(y)
        if self.unique.shape[0] != 2:
            raise ValueError('Just support Two-class Classifier')
        _map = {self.unique[0]: -1, self.unique[1]: 1}
        self.y = np.array([_map[y_i] for y_i in y])
        self.n_samples = X.shape[0]
        self.alpha = np.zeros((self.n_samples))
        self.bias = 0
        self.errorMatrix = np.zeros((self.n_samples, 2))
        entireSet = True
        alphaPairsChanged = 0
        iterCount = 0
        while (((self.max_iter == -1) or (iterCount < self.max_iter)) and ((alphaPairsChanged > 0) or entireSet)):
            alphaPairsChanged = 0
            # 遍历整个数据集
            if entireSet:
                # 遍历整个数据集，找到违背KKT条件的作为第一个参数
                for i in range(self.n_samples):
                    alphaPairsChanged += self.innerLoop(i, y)
                    iterCount += 1
            # 遍历间隔边界点
            else:
                boundList = np.nonzero((self.alpha > 0) & (self.alpha < self.C))[0]
                for i in boundList:
                    alphaPairsChanged += self.innerLoop(i, y)
                    iterCount += 1
            if entireSet:
                entireSet = False
            # 如果遍历间隔边界点，且都符合KKT条件，则遍历整个数据集
            elif alphaPairsChanged == 0:
                entireSet = True
        self.support_ = np.nonzero(self.alpha)[0]
        self.support_vectors_ = X[self.support_]

    def predict(self, X):
        kernelMatrix = self.calcKernelMatrix(self.support_vectors_, X)
        pred = np.dot(self.alpha[self.support_] * self.y[self.support_], kernelMatrix).reshape(-1)
        pred += self.bias
        pred = [1 if pred_i > 0 else -1 for pred_i in pred]
        _map = {-1: self.unique[0], 1: self.unique[1]}
        return [_map[pred_i] for pred_i in pred]
