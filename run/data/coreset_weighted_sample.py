import math
import numpy as np
import random
import bisect

from utils.distance import Euclidean_distance
from .coreset import Coreset


class CoresetWeightedSample(Coreset):
    """
    构建核心集，可以通过两种方式来获取权重

    Examples
    --------
    >>> dataset_util = np.random.uniform(1, 100, (100, 1))
    >>> coreset = CoresetWeightedSample(dataset_util)
    >>> pairset, weight = coreset.sample(100)
    """

    def __init__(self, dataset: np.array):

        """
        该版本的核心集是将整个数据集导入到 Coreset 对象中，然后从 Coreset 对象获取

        :param dataset: dataset_util, array_like
        """
        self.dataset = None
        self.weight = None
        self.mean = None
        self.sample_probability_sum = 0.0
        self.sample_probability = None

        if dataset is not None:
            self.dataset = dataset
            self.n = dataset.shape[0]
            self.get_mean_and_weight()

    def test_mean(self):
        mean = np.mean(self.dataset, axis=0)
        self.mean = mean
        q = np.zeros(self.n)
        sum = 0.0
        self.sample_probability = np.zeros(shape=self.n, dtype=float)
        self.weight = np.zeros(shape=self.n, dtype=float)

        for i in range(self.n):
            q[i] = Euclidean_distance(mean, self.dataset[i, :])  # 中心点到每个点的距离
            self.weight[i] = 1 / q[i]
            sum += (1 / q[i])  # 总距离,所有点到中心的距离之和，定义的是所有数据距离均值点的分布
            self.sample_probability[i] = sum

        self.sample_probability_sum = sum

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
            q[i] = Euclidean_distance(mean, self.dataset[i, :])  # 中心点到每个点的距离
            sum += q[i]  # 总距离,所有点到中心的距离之和，定义的是所有数据距离均值点的分布

        self.sample_probability = q

        for i in range(N):
            q[i] = 0.5 * ((q[i] / sum) + 1 / N)  # (数量越多，减少权重影响)，距离越远，抽中的概率越高
            # q[i] = 0.5 * (q[i] / sum)
            self.sample_probability_sum += q[i]
            self.sample_probability[i] = self.sample_probability_sum

        w = np.zeros(shape=N, dtype=float)

        for m in range(N):
            w[m] = 1 / (q[m] * sum)  # 权重越高，输出权重越少，被抽取的机率越大，给的权重越少
            # w[m] = 1 / q[m] + 1 / sum

        self.mean = mean
        self.weight = w

    def sample(self, n: int, seed) -> (np.array, np.array):
        """
        抽取对象
        """
        ret = []
        random.seed(seed)
        for _ in range(n):
            p = random.uniform(0.0, self.sample_probability_sum)
            idx = bisect.bisect_right(self.sample_probability, p)
            ret.append(idx)
        pairset = []
        weight = []
        # prod = []

        for idx in ret:
            pairset.append(self.dataset[idx])
            weight.append(self.weight[idx])
            # prod.append(self.sample_probability[idx])

        pairset_arr = np.array(pairset)
        weight_arr = np.array(weight)
        # prod_arr = np.array(prod)

        return pairset_arr, weight_arr

    def M(self, k, c, epsilon, delta):
        return c * (self.dataset.shape[0] * k * math.log(k) + math.log(1 / delta, 10)) / (epsilon ** 2)
