import numpy as np
import pandas as pd
import random
from utils.distance import Euclidean_distance
from utils.csvUtil import read_csvfile_to_data_and_label, read_csvfile_to_data_and_label_by_path
import time
import bisect


class Coreset:
    """
    构建核心集，可以通过两种方式来获取权重

    Examples
    --------
    dataset_util = np.random.uniform(1, 100, (100, 1))
    coreset = Coreset(dataset_util)
    mean, weight = coreset.get_mean_and_weight()

    coreset = Coreset(dataset_util)
    weight = coreset.get_batch_weight_by_mean(mean)
    """

    def __init__(self, dataset: np.array):

        """
        initial Coreset class by input dataset_util

        :param dataset: dataset_util, array_like
        """

        self.dataset = dataset
        self.weight = None
        self.mean = None

    def load_dataset_by_name(self, file_name):
        self.dataset_name = file_name
        dataset, labels = read_csvfile_to_data_and_label(file_name)
        self.dataset = dataset
        self.labels = labels

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
            sum += q[i]  # 总距离

        for i in range(N):
            q[i] = 0.5 * ((q[i] / sum) + 1 / N)  # 数量越多，权重越少，距离越近，权重越高

        random.seed()
        w = np.zeros(shape=N, dtype=float)

        for m in range(N):
            w[m] = 100 / (q[m] * sum)  # 权重越高，输出权重越少

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

    def sample_m_points(self, m: int) -> (np.array, np.array):
        """
        抽取对象
        """
        ret = []
        random.seed(time.time())
        for _ in range(m):
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


class C:
    def __init__(self):
        pass

    # def append(self, df: pd.Dataframe):
    #     N = df.shape[0]
    #     mean = df.mean(1)

        # q = np.zeros(N)
        # sum = 0
        #
        # for i in range(N):
        #     q[i] = Euclidean_distance(mean, self.dataset[i, :])  # 中心点到每个点的距离
        #     sum += q[i]  # 总距离
        #
        # for i in range(N):
        #     q[i] = 0.5 * ((q[i] / sum) + 1 / N)  # 数量越多，权重越少，距离越近，权重越高
        #
        # random.seed()
        # w = np.zeros(shape=N, dtype=float)
        #
        # for m in range(N):
        #     w[m] = 100 / (q[m] * sum)  # 权重越高，输出权重越少
        #
        # self.mean = mean
        # self.weight = w
        #
        # return mean, w
