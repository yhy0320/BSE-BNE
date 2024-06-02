import csv
import os
import numpy as np
import random
import pandas as pd

from dataset_util.common import construct_ds_file_path, construct_ds_path
from dataset_util.utils_csv import save_csv_dataset
from dataset_util.utils_rsp import to_rsp
from dataset_util.utils_dataframe import create_default_col


def createCSV(data, label, name):

    dir_name = construct_ds_path(name)
    os.makedirs(dir_name, exist_ok=True)
    filename = construct_ds_file_path(name)
    # 将label数组的形状改变为(N，1)的二维数组，其中N表示label数组中的元素个数
    label = np.reshape(label, (-1, 1))
    # 将两个数组按水平方向进行拼接，返回一个新的数组
    res = np.hstack((data, label))
    df = pd.DataFrame(res)
    create_default_col(df)
    save_csv_dataset(df, name)
    to_rsp(name, 20, 0)

    return filename


def readCSV(name):

    read = open("dataset_util/" + name + ".csv", "r", encoding="utf-8")
    reader = csv.reader(read)
    data = []
    for line in reader:
        data.append(line)

    return data


def writeCSV(dataset, name):

    filename = "dataset_util/" + name + ".csv"
    c = open(filename, "w", encoding="utf-8", newline='')
    writer = csv.writer(c)
    for line in dataset:
        writer.writerow(line)

    c.close()
    return filename


def read_csvfile_to_data_and_label(file_name):

    read = open("../dataset_util/" + file_name + ".csv", "r", encoding="utf-8")
    reader = csv.reader(read)
    data = []
    for line in reader:
        data.append(line)

    Dataset_Label = np.array(data)

    dataset = Dataset_Label[:, :len(Dataset_Label[0]) - 1].astype(float)
    labels = Dataset_Label[:, -1].astype(float).astype(int)

    return dataset, labels


def read_csvfile_to_data_and_label_by_path(file_path):

    read = open(file_path + ".csv", "r", encoding="utf-8")
    reader = csv.reader(read)
    data = []
    for line in reader:
        data.append(line)

    Dataset_Label = np.array(data)

    dataset = Dataset_Label[:, :len(Dataset_Label[0]) - 1].astype(float)
    labels = Dataset_Label[:, -1].astype(float).astype(int)

    return dataset, labels


def read_first_col_label_datase(file_name):

    read = open("../dataset_util/" + file_name + ".csv", "r", encoding="utf-8")
    reader = csv.reader(read)
    data = []
    for line in reader:
        data.append(line)

    Dataset_Label = np.array(data)

    dataset = Dataset_Label[:, :len(Dataset_Label[0]) - 1].astype(float)
    labels = Dataset_Label[:, -1].astype(float).astype(int)

    return dataset, labels


def splitDatasetAndLabel(Dataset_Label: list) -> (np.array, np.array):

    Dataset_Label = np.array(Dataset_Label)

    dataset = Dataset_Label[:, :len(Dataset_Label[0]) - 1].astype(float)
    label = Dataset_Label[:, -1].astype(float).astype(int)

    return dataset, label


def split_to_a_and_b(filename):

    rate = 0.1

    origin = readCSV(filename)

    random.shuffle(origin)

    n = len(origin)

    a = origin[:int(n * rate)]

    b = origin[int(n * rate):]

    a_filename = writeCSV(a, filename + "_a")
    b_filename = writeCSV(b, filename + "_b")

    return a_filename, b_filename


def write_result(result, filename: str):

    filename = "result/" + filename
    c = open(filename, "w", encoding="utf-8", newline='')
    writer = csv.writer(c)
    writer.writerow(result)
    c.close()


def just_get_dataset(filename: str):

    origin = readCSV(filename)
    (dataset, _) = splitDatasetAndLabel(origin)

    return dataset


def sample(dataset, num):

    random.seed()

    n = dataset.shape[0]
    idx_list = [i for i in range(n)]

    s_idx = random.sample(idx_list, num)
    return dataset[s_idx]


if __name__ == '__main__':

    dataset = just_get_dataset("10_100000_2")
    s = sample(dataset, 5000)
    print(s.shape)
