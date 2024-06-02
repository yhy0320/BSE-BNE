import numpy as np
import random
from utils.distance import Euclidean_distance


class CoresetDimison:
    """
    构建核心集，可以通过比较两个数据不同维度下的均值来计算权重

    Examples
    --------
    >>> dataset_util = np.random.uniform(1, 100, (100, 1))
    >>> coreset = Coreset(dataset_util)
    >>> mean, weight = coreset.get_mean_and_weight()

    >>> coreset = Coreset(dataset_util)
    >>> weight = coreset.get_batch_weight_by_mean(mean)
    """

    def __init__(self, dataset: np.array):

        """
        initial Coreset class by input dataset_util

        :param dataset: dataset_util, array_like
        """

        self.dataset = dataset
        self.weight = None
        self.mean = None

    def get_mean_and_weight(self) -> (np.array, np.array):

        """
        返回该核心集的均值(中心点的值)还有所有点对应的权重

        :return mean

        :return weight
        """

        N = self.dataset.shape[0]
        mean = np.mean(self.dataset, axis=0)

        q = np.zeros(N)
        sum = 0

        for i in range(N):
            q[i] = Euclidean_distance(mean, self.dataset[i, :])
            sum += q[i]

        for i in range(N):
            q[i] = 0.5 * ((q[i] / sum) + 1 / N)

        random.seed()
        w = np.zeros(shape=N, dtype=float)

        for m in range(N):
            w[m] = 100 / (q[m] * sum)

        self.mean = mean
        self.weight = w

        return mean, w

    def get_batch_weight_by_mean(self, mean: np.array) -> np.array:

        """
        通过传入均值来计算该核心集对应的权重

        :param mean: 传入的均值
        :return: 权重
        """

        N = self.dataset.shape[0]

        q = np.zeros(N)
        sum = 0

        for i in range(N):
            q[i] = Euclidean_distance(mean, self.dataset[i, :])
            sum += q[i]

        for i in range(N):
            q[i] = 0.5 * ((q[i] / sum) + 1 / N)

        random.seed()
        w = np.zeros(shape=N, dtype=float)

        for m in range(N):
            w[m] = 100 / (q[m] * sum)

        self.mean = mean
        self.weight = w

        return w
