import os
import json
from utils.utils_log import logger
from run.field import *
import numpy as np
from hyperopt import fmin, hp
from dataset_util.dataset_name_consts import *
from dataset_util.common import config_file_path, get_local_exist_dataset_list
from utils.utils_file import get_topic_result_dir_path, drop_failed_prefix, drop_dir_name_failed_prefix
from utils.utils_topic import gen_topic_by_func_and_dataset


CONFIG_FILE_NAME = "conf"
JSON_SUFFIX = ".json"

space_map = {
    ALGORITHM_kmeans: {
        "k": hp.choice('min_samples', range(2, 25)),
    },
    ALGORITHM_gaussian_mixture: {
        "k": hp.choice('min_samples', range(2, 25)),
    },
    ALGORITHM_agglomerative_ward_clustering: {  # ValueError: Expected 2D array, got scalar array instead:
        "k": hp.choice('min_samples', range(2, 25)),
        "connectivity": hp.uniform('xi', 0, 1),
    },
    ALGORITHM_agglomerative_average_clustering: {  # ValueError: Expected 2D array, got scalar array instead:
        "k": hp.choice('min_samples', range(2, 25)),
        "connectivity": hp.uniform('xi', 0, 1),
    },
    ALGORITHM_spectral_clustering: {
        "k": hp.choice('min_samples', range(2, 25)),
    },
    ALGORITHM_dbscan: {
        "eps": hp.uniform('xi', 0, 1),
    },
    ALGORITHM_optics: {
        "min_samples": hp.choice('min_samples', range(3, 10)),
        "xi": hp.uniform('xi', 0, 1),
        "min_cluster_size": hp.uniform('min_cluster_size', 0, 1)
    },
    ALGORITHM_affinity_propagation: {  # 贼妈难运行
        "damping": hp.uniform('xi', 0.5, 1),  # must be >= 0.5.
        "preference": hp.choice('preference', range(-100, 100)),
    },
    ALGORITHM_birch: {
        "k": hp.choice('min_samples', range(2, 25)),
    },
    ALGORITHM_mean_shift: {
        "bandwidth": hp.uniform('bandwidth', 0.5, 1)
    },
    Classify_Nearest_Neighbors: {
        "n_neighbors": hp.choice('n_neighbors', range(2, 10)),
    },
    Classify_Linear_SVM: {
        'C': hp.uniform('C', 0.001, 0.5),
    },
    Classify_RBF_SVM: {
        "gamma": hp.choice('gamma', range(2, 10)),
        "C": hp.uniform('C', 0.001, 0.5),
    },
    Classify_Decision_Tree: {
        "max_depth": hp.choice('max_depth', range(5, 50)),
    },
    Classify_Random_Forest: {
        "max_depth": hp.choice('max_depth', range(5, 50)),
        "n_estimators": hp.choice('n_estimators', range(5, 25)),
        "max_features": hp.choice('max_features', range(1, 10)),
    },
    Classify_Neural_Net: {
        "alpha": hp.uniform('alpha', 0.1, 1),
        "max_iter": hp.choice('n_estimators', [500, 1000, 2000, 5000]),
    },
    Classify_AdaBoost: {
        "learning_rate": hp.uniform('learning_rate', 0.1, 1.0),
    },
    Classify_Naive_Bayes: {
        'var_smoothing': hp.loguniform('var_smoothing', np.log(1e-10), np.log(1e-1)),
    },
}


def config_file_path() -> str:
    return CONFIG_FILE_NAME + JSON_SUFFIX


def load_bne_config(dir_name: str) -> dict:

    if dir_name[-1] != "/":
        dir_name = dir_name + "/"

    file_path = dir_name + config_file_path()
    return load_conf_file(file_path)


def load_conf_file(file_path: str) -> dict:

    """
    获取 dir_name 文件目录下的 conf 文件
    """

    if not os.path.exists(file_path):
        print("我寻思着也没这文件呀")
        logger.error("我寻思着也没这文件呀")
        return {}

    with open(file_path) as f:
        conf = json.load(f)
        return conf


def settle_dataset(run_dataset_list=None) -> list:
    """
    设定要运行什么数据集
    """
    dataset_name_select_map = {}
    if run_dataset_list is None:
        dataset_name_select_map = {
            Dataset_name_5000000_10_10: False,
            Dataset_name_Covtype: False,
            Dataset_name_SUSY: True,
            Dataset_name_PA: False,
            Dataset_name_HIGGS: False,
        }
    elif type(run_dataset_list) == list:
        for ds in run_dataset_list:
            dataset_name_select_map[ds] = True
    elif type(run_dataset_list) == str:
        if run_dataset_list == "all":
            for ds in get_local_exist_dataset_list():
                dataset_name_select_map[ds] = True

    run_dataset_list = []

    for dataset_name, ok in dataset_name_select_map.items():
        if ok:
            run_dataset_list.append(dataset_name)

    return run_dataset_list


def set_dict_with_log(conf: dict, key, value):
    logger.info(f'设置字典 {key} = {value}, 原来的字典为 {conf}')
    conf[key] = value


def save_config(conf: dict):
    """
    保存BNE算法的运行时配置
    """
    with open(conf['dir_name'] + config_file_path(), "w") as f:
        logger.info("record current runtime config start")
        f.write(json.dumps(conf))


def gen_dir_name(conf: dict) -> dict:
    """
    根据 func_name, dataset_name 生成输出文件路径
    """
    topic_name = gen_topic_by_func_and_dataset(conf['function_name'], conf[FIELD_DATASET_NAME])
    dir_name = get_topic_result_dir_path(topic_name)  # 生成执行未成功标记
    conf['dir_name'] = dir_name

    return conf

