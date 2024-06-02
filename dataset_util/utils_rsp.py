import os.path
import random
import shutil
import pandas as pd
# import __init__

from dataset_util.common import *
from dataset_util.consts import *
from dataset_util.utils_csv import load_csv, load_csv_data, save_csv


DS_TYPE_RSP = 1
DS_TYPE_D = 2


def __sample__(population: list, n: int, replace=False) -> list:
    if replace:
        lst = random.sample(population, n)
        for x in lst:
            population.remove(x)
    else:
        lst = random.choices(population, k=n)

    return lst


class D:
    def __init__(self, dataset_name: str, ds_type=DS_TYPE_RSP):
        self.dataset_name = dataset_name
        self.data_list = load_ds_list(dataset_name, ds_type)
        self.selected_list = []
        if ds_type == DS_TYPE_RSP:
            self.dir_name = construct_rsp_path(dataset_name)
        elif ds_type == DS_TYPE_D:
            self.dir_name = construct_d_path(dataset_name)
        else:
            assert False, "啥玩意"

    def sample(self, n: int, replace=False) -> list:
        sample = __sample__(self.data_list, n, replace)
        if replace:
            for r in sample:
                self.selected_list.append(r)

        return sample

    def selected_file_2_dataframe(self) -> pd.DataFrame:
        return rsp_file_2_dataframe(self.selected_list, self.dir_name)

    def sample_dataframe(self, n: int, replace=False) -> pd.DataFrame:
        file_list = self.sample(n, replace)
        return rsp_file_2_dataframe(file_list, self.dir_name)


def rsp_file_2_dataframe(file_list: list, dir_path: str) -> pd.DataFrame:
    rsp_list = []
    for block_file_name in file_list:
        rsp = load_csv(dir_path + block_file_name)
        rsp_list.append(rsp)
    # concat用于沿着指定的轴(默认为行轴)将多个DataFrame或Series对象拼接在一起
    sample = pd.concat(rsp_list, axis=0, ignore_index=True)
    return sample


def load_ds_list(dataset_name: str, ds_type: int) -> list:

    if ds_type == DS_TYPE_RSP:
        dir_path = construct_rsp_path(dataset_name)
    elif ds_type == DS_TYPE_D:
        dir_path = construct_d_path(dataset_name)
    else:
        assert False, f"error data type = {ds_type}"

    ds_list = os.listdir(dir_path)
    ds_list = list(filter(lambda x: x.startswith('rsp_'), ds_list))
    return ds_list


def select_rsp_list(dataset_name: str, rsp_block_num: int) -> (list, int):
    """
    获取到对应 dataset_name 的 RSP 数据块列表

    :param dataset_name: 数据集名称
    :param rsp_block_num: RSP数据块数量

    RETURN
    --------
    rsp_list: rsp数据块列表
    n: rsp数据块总数
    """

    assert rsp_block_num > 0 or rsp_block_num == -1, "你要取啥啊"

    dir_path = construct_rsp_path(dataset_name)
    rsp_list = os.listdir(dir_path)
    rsp_list = list(filter(lambda x: x.startswith('rsp_'), rsp_list))
    n = len(rsp_list)

    assert n > 0, "这都没RSP块啊"

    if rsp_block_num == -1:
        rsp_block_num = n

    sample_list = random.sample(rsp_list, rsp_block_num)
    rsp_list = []
    for block_file_name in sample_list:
        rsp = load_csv(dir_path + block_file_name)
        rsp_list.append(rsp)

    return rsp_list, n


def select_rsp_merged_block(dataset_name: str, rsp_block_num: int) -> pd.DataFrame:
    """
    获取到对应 dataset_name 的 RSP 数据块列表

    :param dataset_name: 数据集名称
    :param rsp_block_num: RSP数据块数量

    :return 合并后的rsp数据块
    """

    rsp_list, _ = select_rsp_list(dataset_name, rsp_block_num)
    sample = pd.concat(rsp_list, axis=0, ignore_index=True)
    return sample


def to_rsp(dataset_name: str, block_num: int, block_size: int):

    df = load_csv_data(dataset_name)
    rsp_blocks, block_size, block_num = quantity_split(df, block_num, block_size)
    dir_path = construct_rsp_path(dataset_name)
    os.makedirs(dir_path, exist_ok=True)
    _save_rsp(rsp_blocks, dir_path)
    dataset_conf = load_dataset_conf(dataset_name)
    save_rsp_config(dir_path, block_num, block_size, dataset_name, dataset_conf)


def to_d(dataset_name: str, block_num: int, block_size: int):

    df = load_csv_data(dataset_name)
    rsp_blocks, block_size, block_num = quantity_split(df, block_num, block_size)
    dir_path = construct_d_path(dataset_name)
    os.makedirs(dir_path, exist_ok=True)
    _save_rsp(rsp_blocks, dir_path)
    dataset_conf = load_dataset_conf(dataset_name)
    save_rsp_config(dir_path, block_num, block_size, dataset_name, dataset_conf)


def _save_rsp(rsp_blocks: list, dir_path: str):

    idx = 1
    for rsp in rsp_blocks:
        save_csv(dir_path + RSP_BLOCK_FILE_PREFIX + f"{idx}.csv", rsp)
        idx += 1


def load_rsp_config(dataset_name: str) -> dict:

    dir_name = construct_rsp_path(dataset_name)
    with open(dir_name + config_file_path()) as f:
        conf = json.load(f)

    return conf


def load_d_config(dataset_name: str) -> dict:

    dir_name = construct_d_path(dataset_name)
    with open(dir_name + config_file_path()) as f:
        conf = json.load(f)

    return conf


def save_rsp_config(dir_name: str, block_num: int, block_size: int, dataset_name: str, dataset_conf: dict):

    conf = {
        'dataset_name': dataset_name,
        'block_num': block_num,
        'block_size': block_size,
        "instance": dataset_conf['instance'],
        "cluster": dataset_conf['cluster'],
        "dim": dataset_conf['dim'],
    }

    with open(dir_name + CONFIG_FILE_NAME + JSON_SUFFIX, "w") as f:
        # json.dumps()用于将Python对象转换成JSON格式的字符串
        f.write(json.dumps(conf))


def get_rsp_block_num(dataset_name: str) -> int:
    dir_path = construct_rsp_path(dataset_name)
    rsp_list = os.listdir(dir_path)
    rsp_list = list(filter(lambda x: x.startswith('rsp_'), rsp_list))
    n = len(rsp_list)
    return n


def select_rsp_block_from_d(dataset_name: str, block_num: int, block_size=2):
    """
    block_size表示多少个d数据集，默认值是2个
    """
    _total_d = block_num * block_size


if __name__ == '__main__':
    # for ds in get_local_exist_dataset_list():
    #     dir_path = construct_d_path(ds)
    #     if os.path.exists(dir_path):
    #         shutil.rmtree(dir_path)
    #     print(f'>>> {ds}')
    to_d(Dataset_name_HIGGS, 0, 1000)

