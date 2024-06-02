import random
import numpy as np
import matplotlib.pyplot as plt

from utils.csvUtil import read_csvfile_to_data_and_label, read_csvfile_to_data_and_label_by_path


class Dataset:

    def __init__(self, dataset=None, labels=None, dataset_name="", seed=0):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.labels = labels
        self.seed = seed

    def load_dataset_by_name(self, file_name):
        self.dataset_name = file_name
        dataset, labels = read_csvfile_to_data_and_label(file_name)
        self.dataset = dataset
        self.labels = labels

    def load_dataset_by_path(self, file_path):
        self.dataset_name = file_path.split("/")[-1]
        dataset, labels = read_csvfile_to_data_and_label_by_path(file_path)
        self.dataset = dataset
        self.labels = labels

    def draw_dataset(self, auto_draw=True):
        plt.scatter(self.dataset[:, 0], self.dataset[:, 1])
        if auto_draw:
            plt.show()

    def sample(self, n: int) -> (np.array, np.array):
        random.seed(self.seed)
        sample_idx = random.sample(range(self.dataset.shape[0]), n)
        return self.dataset[sample_idx], self.labels[sample_idx]

    def gen_sub_dataset(self, n: int):
        subset = Dataset()

        dataset, labels = self.sample(n)
        subset.dataset = dataset
        subset.labels = labels
        subset.dataset_name = self.dataset_name + "[subset][" + str(n) + "]"

        return subset

    def get_data_list(self):
        return self.dataset.tolist()

    def loadtxt(self, file_name: str):
        self.dataset_name = file_name
        arr = np.loadtxt("../dataset_util/" + file_name + ".csv")
        self.dataset = arr[:, :arr.shape[1] - 1]
        self.labels = arr[:, arr.shape[1] - 1]
