"""这个文件夹主要是用来维护数据装载器的公共方法"""
import json
import os
import pandas as pd
# import __init__
from utils import get_base_path
import math
from dataset_util.utils_dataframe import shuffle
from dataset_util.dataset_name_consts import *
from dataset_util.consts import *


def construct_ds_path(dataset_name: str) -> str:
    # /data_split/ds/{dataset_name}/
    base_path = get_base_path()
    return base_path + DATASET_PATH + dataset_name + "/"


def ds_file_path() -> str:
    # ds.csv
    return DATASET_FILE_NAME + CSV_SUFFIX


def test_file_path() -> str:
    # test.csv
    return TEST_FILE_NAME + CSV_SUFFIX


def config_file_path() -> str:
    # conf.json
    return CONFIG_FILE_NAME + JSON_SUFFIX


def construct_ds_file_path(dataset_name: str) -> str:
    # /data_split/ds/{dataset_name}/ds.csv
    return construct_ds_path(dataset_name) + DATASET_FILE_NAME + CSV_SUFFIX


def construct_d_path(dataset_name: str) -> str:
    # /data_split/ds/{dataset_name}/d/
    return construct_ds_path(dataset_name) + D_PATH


def construct_rsp_path(dataset_name: str) -> str:
    # /data_split/ds/{dataset_name}/rsp/
    return construct_ds_path(dataset_name) + RSP_PATH


def construct_test_path(dataset_name: str) -> str:
    # /data_split/ds/{dataset_name}/tests/
    return construct_ds_path(dataset_name) + TEST_PATH


def split_data_and_label(raw: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    col = raw.shape[1]
    # iloc根据位置(整数索引)选择子集，第一个参数表示要选择的行的位置，第二个参数表示要选择的列的位置
    return raw.iloc[:, :col - 1], raw.iloc[:, -1]


def quantity_split(source_data: pd.DataFrame, block_num: int, block_size: int) -> (list, int):
    """
    dataframe拆分

    :param source_data: 源数据集
    :param block_num: 需要拆分为多少份
    :param block_size: 每个块的数量

    :return 返回值："分块"，"每块的记录数"
    """
    n = source_data.shape[0]
    block_num, block_size = fix_num_and_size(n, block_num, block_size)

    ret = []
    # 打乱数据
    source_data = shuffle(source_data)

    for i in range(block_num):
        df = source_data[block_size * i: block_size * (i + 1)]
        ret.append(df)

    return ret, block_size, block_num


def fix_num_and_size(n: int, block_num: int, block_size: int) -> (int, int):
    if block_num != 0 and block_size == 0:
        block_size = math.ceil(n / block_num)
        return block_num, block_size

    if block_num == 0 and block_size != 0:
        block_num = math.ceil(n / block_size)
        return block_num, block_size

    assert False, f"不能两个同时不为0，或同时为0, block_num: {block_num}, block_size: {block_size}"


def load_dataset_conf(dataset_name: str) -> dict:
    """
    获取数据集的 conf 数据
    """

    dir_name = construct_ds_path(dataset_name)
    conf = load_conf(dir_name)

    return conf


def load_conf(dir_name: str) -> dict:
    """
    获取 dir_name 文件目录下的 conf 文件
    """

    if dir_name[-1] != "/":
        dir_name = dir_name + "/"

    assert os.path.exists(dir_name + config_file_path()), "我寻思着也没这文件呀"
    # /data_split/ds/{dataset_name}/conf.json
    with open(dir_name + config_file_path()) as f:
        conf = json.load(f)


    return conf


def get_local_exist_dataset_info() -> pd.DataFrame:
    """
    获取本地存在的所有数据集信息
    """

    ds_list = get_local_exist_dataset_list()
    dir_name = get_base_path() + DATASET_PATH

    df = pd.DataFrame(columns=['dataset_name', 'instance', 'cluster', 'dim'])

    for dataset_name in ds_list:
        conf = load_conf(dir_name + dataset_name + "/")
        # print(conf)
        df = df.append({
            'dataset_name': dataset_name,
            'instance': conf['instance'],
            'cluster': conf['cluster'],
            'dim': conf['dim'],
        }, ignore_index=True)

    return df


def get_local_exist_dataset_list(exclude_big_data=False) -> list:
    dir_name = get_base_path() + DATASET_PATH
    ds_list = os.listdir(dir_name)
    if exclude_big_data:
        ds_list.remove(Dataset_name_2000000_10_10_dataset)
        ds_list.remove(Dataset_name_HIGGS)
        ds_list.remove(Dataset_name_SUSY)
        ds_list.remove(Dataset_name_1000000_200_2_dataset)

    return ds_list


def get_local_exist_dataset_cite():
    dir_name = get_base_path() + DATASET_PATH
    ds_list = os.listdir(dir_name)

    df = pd.DataFrame(columns=['dataset_name', 'instance', 'cluster', 'dim'])

    for dataset_name in ds_list:
        conf = load_conf(dir_name + dataset_name + "/")
        if "cite" in conf:
            print(conf['cite'])

    return df


def gen_local_dataset_consts_file():
    dir_name = get_base_path() + DATASET_PATH
    ds_list = os.listdir(dir_name)

    file_content = ""
    for ds in ds_list:
        file_content += f'Dataset_name_{ds} = "{ds}"\n'

    with open("./dataset_name_consts.py", 'w+') as f:
        f.write(file_content)


def sample_with_split_data_and_label(raw: pd.DataFrame, num: int) -> (pd.DataFrame, pd.DataFrame):
    col = raw.shape[1]
    # s = raw.sample(n=20000, replace=True, random_state=666)
    s = raw.sample(n=num, replace=True)
    return s.iloc[:, :col - 1], s.iloc[:, -1]
