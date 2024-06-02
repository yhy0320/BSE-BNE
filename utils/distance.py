import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def Euclidean_distance(p: np.array, q: np.array) -> float:
    # return np.sqrt(np.sum((p - q) ** 2))
    tol = p - q
    return np.sqrt(np.einsum("i,i->", tol, tol))


# 这个函数计算出来不是Mahalanobis_distance`
def Mahalanobis_distance(p: np.array, q: np.array):
    ret = 0
    for i in range(p.shape[0]):
        ret += abs(p[i] - q[i])
    return ret


def x_with_centers_euli(x: np.array, C: np.array):
    ret = 0.0
    for c in C:
        dis = Euclidean_distance(x, c)
        if dis < ret:
            ret = dis
    return ret


def find_closest_idx(x, centers):
    ret = 0
    bigger = float('inf')
    for i in range(centers.shape[0]):
        dis = Euclidean_distance(x, centers[i])
        if dis < bigger:
            bigger = dis
            ret = i
    return ret


def cal_martix(X):
    n = X.shape[0]
    ret = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dis = Euclidean_distance(X[i], X[j])
            ret[i][j] = dis
            ret[j][i] = dis

    return ret


def find_all_closest_idx(X, centers):
    center_half_distances = cal_martix(centers) / 2

    n = X.shape[0]
    n_cluster = centers.shape[0]
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        min_dis = Euclidean_distance(X[i], centers[0])
        minLabel = 0
        for j in range(1, n_cluster):
            if min_dis > center_half_distances[minLabel][j]:
                thisDist = Euclidean_distance(X[i], centers[j])
                if thisDist < min_dis:
                    min_dis = thisDist
                    minLabel = j

        labels[i] = minLabel

    return labels
