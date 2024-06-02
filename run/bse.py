import json
import math
import time
import uuid
from datetime import datetime

import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans
from tqdm import tqdm

import dataset_util.dataset_name_consts as ds_const
import run.model.runtime as rt
import utils.utils_dict
from bne_config import space_map
from dataset_util.common import load_dataset_conf
from dataset_util.dataset_name_consts import *
from dataset_util.utils_csv import load_csv_data
from dataset_util.utils_rsp import select_rsp_merged_block
from run.data.coreset import Coreset
from run.field import *
from utils.distance import Euclidean_distance
from utils.utils_log import logger


def cal_lower_bound(d, k, delta=0.1, eps=0.1):
    """
    计算样本数量下界
    :param d: 数据集维度
    :param k: 簇数目
    :param delta: 置信度，算法结果与真实结果之间的差距不超过eps的概率至少为1-delta，默认为0.1
    :param eps: 聚类误差，eps越小样本质量越好，默认为0.1
    """
    return (d * k * math.log(k) + math.log(1 / delta)) / (eps ** 2)


def construct_core_set(dataset, m):
    """
    给每个点一个相同的权重，然后利用权重抽样构建核心集
    :param dataset: 数据集
    :param m: 核心集大小
    """
    # 计算每个数据点的权重
    weights = np.zeros(len(dataset))
    for i, x in enumerate(dataset):
        weights[i] = 1 / len(dataset)

    # 从数据集中随机抽样一定数量的数据点
    indices = np.random.choice(len(dataset), size=m, replace=False, p=weights)

    # 构建核心集
    coreset = dataset[indices]

    return coreset, indices


def sampleDataset(dataset, num):
    """
    从数据集中抽取指定数量的数据
    :param dataset: 数据集
    ：param num: 抽取数量
    """
    indices = np.random.choice(len(dataset), size=num, replace=False)
    samples = dataset[indices]
    return samples, indices


def get_lightweight_core_set(dataset, m):
    """
    构建一个轻量级核心集
    :param dataset: 数据集
    :param m: 轻量级核心集大小
    """
    # calculate the mean of all data points
    mu = np.mean(dataset, axis=0)

    # first term in prob distribution
    a = 1 / (2 * len(dataset))

    # denominator in second term of prob distribution
    sum_dsq = np.sum((LA.norm(mu - dataset, axis=1)) ** 2)

    # 添加容错机制，避免除以0
    if sum_dsq == 0:
        sum_dsq = 1e-6

    # assign probability to each point
    q = []
    for data in dataset:
        dsq = (LA.norm(mu - data)) ** 2
        q.append(a + dsq / (2 * sum_dsq))

    # sample m points from this distribution
    samples = np.random.choice(dataset.shape[0], size=m, replace=False, p=q)

    return dataset[samples]


def weighted_quantization_error(dataset: np.array, w: np.array, center: np.array) -> float:
    """
    计算某组数据对于某组权重和中心点的量化误差
    :param dataset: 目标数据集
    :param w: 数据集对应的权重集合
    :param center: 目标中心点
    :return: 量化误差
    """
    quantization_error = 0

    for i, x in enumerate(dataset):
        cur_error = float('inf')
        for c in center:
            cur_error = min(Euclidean_distance(x, c), cur_error)
        # 这里乘了100，所以实际量化误差是计算结果的1/100
        quantization_error += (w[i] * 100 * cur_error)

    return quantization_error


def create_bse_config() -> dict:
    """
    创建BSE算法运行时的配置
    """
    global global_algorithm, global_dataset_name, batch_run_id

    conf = {
        'dataset_name': global_dataset_name,
        "algorithm": global_algorithm,
        'm': 0,
        'seed': int(time.time()),
        'run_id': uuid.uuid1().__str__(),
        'function_name': bse_new.__name__,
        'population_run': True,
        'ratio': 0.3,
        'threshold': 19}

    if batch_run_id != "":
        conf['batch_run_id'] = batch_run_id
    else:
        conf['batch_run_id'] = ""

    return conf


def save_config(conf: dict):
    """
    保存BSE算法运行时的配置
    """
    rt.insert({
        'run_id': conf['run_id'],
        'config': json.dumps(conf),
        'batch_run_id': conf['batch_run_id'],
        'seed': conf['seed'],
    })
    logger.info(f'插入配置到mysql数据库中, run_id = {conf["run_id"]}, conf = {conf}')


def get_runtime_config(run_id: str) -> dict:
    df = rt.query({
        'run_id': run_id,
    })
    if df.shape[0] <= 0:
        return {}

    conf = json.loads(df.loc[0, 'config'])
    return conf


def get_cluster_center(data, weight, k, seed):
    """
    获取数据集的中心点
    """
    logger.info(f'data shape: {data.shape}, weight shape: {weight.shape}, k: {k}, seed: {seed}')
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(data, np.multiply(weight, 100))
    return kmeans.cluster_centers_


def bse_old(conf: dict):
    """
    NOTE: 该代码仅仅就是运行程序，保存结果，不对流程有任何的判断

    流程：
    1. 抽取两个对称样本
    2. 创建两个样本的核心集
    3. 比较在不同 k 值下，两个核心集的加权量化误差

    :param conf: 运行时配置文件
    """
    dataset_name = conf['dataset_name']
    origin_dataset_conf = load_dataset_conf(dataset_name)
    n = origin_dataset_conf['instance']
    iter_num = 20

    pbar = tqdm(range(1, iter_num + 1), ncols=100)

    k = conf[FIELD_K]

    for i in pbar:
        pbar.set_description(f'd num: {i}')

        # step 1: 获取指定数量的 两个对称集合
        pair_set_0 = select_rsp_merged_block(dataset_name, i)  # , DS_TYPE_D)
        pair_set_1 = select_rsp_merged_block(dataset_name, i)  # , DS_TYPE_D)
        logger.info(f'sample pair_set 0, shape = {pair_set_0.shape},  pair_set 1, shape = {pair_set_1.shape}')

        # step 2: 获取对称集 1 的 中心， 权重
        corset_0 = Coreset(np.array(pair_set_0))
        (mean, w_0) = corset_0.get_mean_and_weight()
        logger.info(f'get core set 0 mean: {mean}, weight length: {len(w_0)}')

        # step 3: 把对称集 1 获得的中心放到对称集 2 中计算对称集 2 的权重
        corset_1 = Coreset(np.array(pair_set_1))
        w_1 = corset_1.get_batch_weight_by_mean(mean)
        logger.info(f'get core set 0 weight length: {len(w_1)}')

        # step 4: 通过 聚类 算法算出对称集 1 的中心点
        # line 8
        pair_set_0_center = get_cluster_center(pair_set_0, w_0, k, conf['seed'])
        logger.info(f'get pair set 0 center shape: {pair_set_0_center.shape}')

        # step 5: 通过对称集 1 的中心点，计算两个对称集的量化误差
        # line 9 ~ line 19
        p_0_qe = weighted_quantization_error(np.array(pair_set_0), w_0, pair_set_0_center) / pair_set_0.shape[0]
        p_1_qe = weighted_quantization_error(np.array(pair_set_1), w_1, pair_set_0_center) / pair_set_1.shape[0]

        logger.info(f'get p_0_qe: {p_0_qe}, p_1_qe: {p_1_qe}')

        logger.info(f'insert new result, iter = {i}')


def bse_new(conf: dict):
    """
    下面的代码是根据论文中的伪代码写的

    流程：
    1. 抽取两个对称样本
    2. 创建两个样本的核心集
    3. 比较两个核心集的加权量化误差，如果小于threshold，结束迭代，否则一直迭代

    :param conf: 运行时配置文件
    """

    print("=====现在开始一轮新的抽样！=====")

    dataset_name = conf['dataset_name']
    ratio = conf['ratio']
    threshold = conf['threshold']

    origin_dataset_conf = load_dataset_conf(dataset_name)
    d = origin_dataset_conf['dim']
    k = origin_dataset_conf['cluster']
    m = math.ceil(cal_lower_bound(d, k, 0.1, threshold / 100))
    conf['m'] = m

    print('k is:', k, ', d is:', d, ', the lower bound m is:', m)

    dataset_1 = load_csv_data(dataset_name)
    dataset_1 = np.array(dataset_1)
    dataset_2 = load_csv_data(dataset_name)
    dataset_2 = np.array(dataset_2)

    # step 1: 根据计算出来的lower bound，获取指定数量的 两个对称集合
    # line 1~2
    C_1, indices_1 = construct_core_set(dataset_1, m)
    C_2, indices_2 = construct_core_set(dataset_2, m)

    logger.info(f'sample C_1, shape = {C_1.shape},  C_2, shape = {C_2.shape}')

    # 将已经抽取的数据从原数据中去除
    dataset_1 = np.delete(dataset_1, indices_1, axis=0)
    dataset_2 = np.delete(dataset_2, indices_2, axis=0)

    # line 3
    diff_qe = math.inf
    count = 1  # 用于统计迭代次数

    while diff_qe > threshold:
        # step 2: 根据给定的ratio抽取指定数量的集合，然后获取对应lightweight coreset
        # line 5
        # B_1 = sample_csv_data(dataset_name, math.ceil(ratio * m))
        # B_2 = sample_csv_data(dataset_name, math.ceil(ratio * m))
        B_1, indices_1 = sampleDataset(dataset_1, math.ceil(ratio * m))
        B_2, indices_2 = sampleDataset(dataset_2, math.ceil(ratio * m))

        # 将已经抽取的数据从原数据中去除
        dataset_1 = np.delete(dataset_1, indices_1, axis=0)
        dataset_2 = np.delete(dataset_2, indices_2, axis=0)

        # line 6
        cB_1 = get_lightweight_core_set(B_1, math.ceil(ratio * m * 0.5))
        cB_2 = get_lightweight_core_set(B_2, math.ceil(ratio * m * 0.5))
        # if dataset_name == 'Covtype':
        #     cB_1 = get_lightweight_core_set(B_1, math.ceil(ratio * m * 0.2))
        #     cB_2 = get_lightweight_core_set(B_2, math.ceil(ratio * m * 0.2))

        # line 7
        C_1 = np.concatenate((C_1, cB_1), axis=0)
        C_2 = np.concatenate((C_2, cB_2), axis=0)

        # step 3: 获取对称集 1 的 中心，权重
        corset_1 = Coreset(C_1)
        (mean, w_1) = corset_1.get_mean_and_weight()
        logger.info(f'get core set 1 mean: {mean}, weight length: {len(w_1)}')

        # step 4: 把对称集 1 获得的中心放到对称集 2 中计算对称集 2 的权重
        corset_2 = Coreset(C_2)
        w_2 = corset_2.get_batch_weight_by_mean(mean)
        logger.info(f'get core set 2 weight length: {len(w_2)}')

        # step 5: 通过 聚类 算法算出对称集 1 的中心点
        # line 8
        C_1_center = get_cluster_center(C_1, w_1, k, conf['seed'])
        logger.info(f'get C_1 center shape: {C_1_center.shape}')

        # step 6: 通过对称集 1 的中心点，计算两个对称集的量化误差
        # line 9 ~ line 19
        qe_1 = weighted_quantization_error(C_1, w_1, C_1_center) / C_1.shape[0]
        qe_2 = weighted_quantization_error(C_2, w_2, C_1_center) / C_2.shape[0]

        logger.info(f'get p_0_qe: {qe_1}, p_1_qe: {qe_2}')
        logger.info(f'insert new result, iter = {count}')

        diff_qe = abs(qe_1 - qe_2)

        print('当前抽取样本数量为:', C_1.shape[0], '量化误差为:', diff_qe)
        count += 1

    return C_1.shape[0]


def objective(param, iterationNum):
    config = create_bse_config()
    config = utils.utils_dict.merge_dict(param, config)
    config['threshold'] = global_threshold

    recordNum = []
    for _ in range(iterationNum):
        recordNum.append(bse_new(config))
    return recordNum


def bse_tune(max_evals=10):
    global global_algorithm
    space = space_map[global_algorithm]
    """
    fmin()用于使用给定的搜索算法在超参数搜索空间中寻找最优的超参数组合
    fn 评估函数
    space 搜索空间
    max_evals 最大评估次数
    """
    # fmin(fn=objective, space=space, max_evals=max_evals, show_progressbar=False)
    return objective(space, max_evals)


# 数据集执行的全局变量
batch_run_id = ''
global_dataset_name = ds_const.Dataset_name_5000000_10_10
global_algorithm = ALGORITHM_kmeans
global_threshold = 10

if __name__ == '__main__':
    thresholdList = [20, 19]
    dsList = [Dataset_name_5000000_10_10]
    loopNumber = 2
    file = open("bse_result.txt", "a")
    cur_time = datetime.now()
    file.write("=========================BSE算法开始运行==========================\n")
    file.write("开始运行时间：{}\n".format(cur_time))
    for threshold in thresholdList:
        print(">>> 当前threshold为：{}".format(threshold))
        file.write(">>> 当前threshold为：{}\n".format(threshold))
        for ds in dsList:
            print(f'>>> 当前运行数据集为: {ds}')
            file.write(f'>>> 数据集：{ds}\n')
            global_dataset_name = ds
            global_threshold = threshold
            start_time = time.time()  # 记录开始时间
            recordNum = bse_tune(loopNumber)
            end_time = time.time()  # 记录结束时间

            avg_time = (end_time - start_time) / loopNumber
            print("运行平均时间为：{}秒".format(avg_time))
            file.write(">>> 运行平均时间为：{}秒\n".format(avg_time))
            avg_num = math.ceil(sum(recordNum) / loopNumber)
            print("样本数目为：{}".format(avg_num))
            file.write(">>> 样本数目为：{}\n".format(avg_num))
    file.write("=========================BSE算法运行结束==========================\n\n")
    file.close()
