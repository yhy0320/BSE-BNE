import math
import random
import time
import uuid

from scipy.stats import t
from tqdm import tqdm

import dataset_util.dataset_name_consts as ds_const
from bne_config import *
from dataset_util import utils_csv
from dataset_util.common import split_data_and_label
from dataset_util.utils_rsp import D, DS_TYPE_D
from dataset_util.utils_rsp import load_d_config
from run.algorithm_model import get_model_type
from run.bne_ana import params_format
from run.classifier import run_classify, classifiers_map
from run.clustering import run_clustering, default_clustering_conf
from run.consts import *
from run.model.global_result import run_global_result
from utils.utils_dict import extract_special_config


def compare_increase(last_score, score, threshold=None, percent=None) -> bool:
    """
    :return 是否继续, True 继续执行， False 取消执行
    """
    logger.info(f'last_score = {last_score}, score = {score}, threshold = {threshold}, percent = {percent}')

    if threshold is not None:
        return abs(score - last_score) > threshold

    if percent is not None:
        return (abs(score - last_score) / score) > percent

    return True


def calculate_t(dataset_name, delta=0.1):
    data = utils_csv.load_csv_data(dataset_name)

    # 删除最后一列数据(label)
    data = data.drop(data.columns[-1], axis=1)

    # 将DataFrame数据转换为array数据
    array = data.values
    means = []
    for i in range(array.shape[0]):
        mean = np.mean(array[i], axis=0)
        means.append(mean)

    # 计算置信区间的上下限
    df = data.shape[1] - 1
    t_value = t.ppf(1 - delta, df)
    std_error = np.std(means, axis=0) / np.sqrt(data.shape[0])
    lower = np.mean(means, axis=0) - t_value * std_error
    upper = np.mean(means, axis=0) + t_value * std_error

    # 计算误差下界
    error_bound = np.linalg.norm(upper - lower) / np.linalg.norm(np.mean(means, axis=0))

    return error_bound


def calculate_lower_bound(t, tao, delta=0.001):
    r = ((2 * delta ** 2) / t ** 2) * math.log(2 / tao)
    return math.ceil(r)


def bne(conf: dict):
    if 'population_run' in conf and conf['population_run']:
        run_global_result(conf)

    return iter_run_algorithm_new(conf)


def iter_run_algorithm_old(conf: dict):
    """
    迭代的运行不同RSP数据块下的结果
    但是
    这个是
    分类算法芜湖！
    """
    logger.info(f'运行分类算法测试逻辑')
    d_conf = load_d_config(conf['dataset_name'])  # 获取RSP数据块配置

    D_PER_R = conf['d_per_r']
    pbar = tqdm(range(conf[FIELD_R], min(50, d_conf['block_num'] // D_PER_R + 1)))

    last_score = math.inf
    score = math.inf

    satisfied_time = 0

    rsp = D(conf['dataset_name'], DS_TYPE_D)

    print(conf)

    for num in pbar:
        pbar.set_description(f'{params_format(extract_special_config(conf, conf["runtime_key"]))}[{conf["algorithm"]}];'
                             f'rsp num: {num}')

        logger.info(f"run {num} rsp_block result")
        rsp.sample(1 * D_PER_R, replace=True)
        ds, label = split_data_and_label(rsp.selected_file_2_dataframe())
        logger.info(f"load rsp block, dataset shape =  {ds.shape}")

        if conf[FIELD_MODEL_TYPE] == CONSTS_MODEL_TYPE_CLASSIFY:
            metrics, cost_time = run_classify(ds, label, conf)
        elif conf[FIELD_MODEL_TYPE] == CONSTS_MODEL_TYPE_CLUSTERING:
            metrics, cost_time = run_clustering(ds, label, conf)
        else:
            assert False, "这啥算法啊"

        # 将本次实验结果插入数据库中
        # new_conf = conf.copy()
        # new_conf['instance'] = ds.shape[0]
        # new_conf['dimension'] = ds.shape[1]
        # new_conf['cost_time'] = cost_time
        # params_with_algorithm = list(conf["runtime_key"]).copy()
        # if 'algorithm' not in params_with_algorithm:
        #     params_with_algorithm.append('algorithm')
        # new_conf['params'] = order_dumps(extract_special_config(conf, params_with_algorithm))
        # new_conf['metrics'] = json.dumps(metrics)
        # new_conf['file_path'] = conf['dir_name']
        # new_conf['batch_run_id'] = conf['batch_run_id']
        # new_conf['config'] = json.dumps(conf)

        # logger.info(f"insert runtime result into mysql, result = {new_conf}")
        # insert_bne_result(new_conf)

        # 写入抽样运行和总数据集运行的指标
        logger.info(f"insert runtime result into excel")


def iter_run_algorithm_new(conf: dict):
    """
    迭代的运行不同RSP数据块下的结果
    但是
    这个是
    分类算法芜湖！
    """
    logger.info(f'运行分类算法测试逻辑')

    threshold = conf['threshold']
    rsp = D(conf['dataset_name'], DS_TYPE_D)

    # line 1~3
    n = conf['r'] - 1
    last_score = math.inf
    score = math.inf

    # line 4
    rsp.sample(n, replace=True)
    # ds, label = split_data_and_label(rsp.selected_file_2_dataframe())

    satisfied_time = 0

    while abs(last_score - score) / last_score > threshold:
        # line 6~7
        rsp.sample(1, replace=True)
        ds, label = split_data_and_label(rsp.selected_file_2_dataframe())
        logger.info(f"load rsp block, dataset shape =  {ds.shape}")
        n = n + 1

        if conf[FIELD_MODEL_TYPE] == CONSTS_MODEL_TYPE_CLASSIFY:
            metrics, cost_time = run_classify(ds, label, conf)
        elif conf[FIELD_MODEL_TYPE] == CONSTS_MODEL_TYPE_CLUSTERING:
            metrics, cost_time = run_clustering(ds, label, conf)
        else:
            assert False, "这啥算法啊"

        # line 8~9
        if conf[FIELD_MODEL_TYPE] == CONSTS_MODEL_TYPE_CLASSIFY:
            last_score = score
            score = metrics['score']

        # 判断是否需要跳过迭代
        if 'skip_iter_time' in conf and cost_time > conf['skip_iter_time']:
            logger.info(
                f'跳过本次耗时迭代流程, cost_time = {cost_time}, skip_iter_time = {conf["skip_iter_time"]}')
            break

        # 增加BNE停止算法，目前只支持分类算法
        if 'stop_iter_score' in conf:
            stop_iter_score = conf['stop_iter_score']

            if conf[FIELD_MODEL_TYPE] == CONSTS_MODEL_TYPE_CLASSIFY:
                last_score = score
                score = metrics['score']

            if 'threshold' in stop_iter_score:
                if compare_increase(last_score, score, threshold=stop_iter_score['threshold']):
                    logger.info(f'不满足暂停条件, last_score = {last_score}, score = {score}')
                    satisfied_time = 0
                else:
                    logger.info(f'满足暂停条件, last_score = {last_score}, score = {score}')
                    satisfied_time += 1

            if 'percent' in stop_iter_score:
                if compare_increase(last_score, score, percent=stop_iter_score['percent']):
                    logger.info(f'不满足暂停条件, last_score = {last_score}, score = {score}')
                    satisfied_time = 0
                else:
                    logger.info(f'满足暂停条件, last_score = {last_score}, score = {score}')
                    satisfied_time += 1

            if satisfied_time == 2:
                break
            else:
                continue

        # 写入抽样运行和总数据集运行的指标
        logger.info(f"insert runtime result into excel")

    return n


def get_bne_config() -> dict:
    global global_algorithm, global_dataset_name, batch_run_id

    print(global_dataset_name)
    conf = {
        FIELD_DATASET_NAME: global_dataset_name,
        FIELD_ALGORITHM: global_algorithm,
        FIELD_R: 1,
        'seed': int(time.time()),
        'run_id': uuid.uuid1().__str__(),
        'function_name': bne.__name__,
        'population_run': True,
        'd_per_r': 2,
        'threshold': 0.8
    }

    if batch_run_id is not None:
        conf['batch_run_id'] = batch_run_id
    else:
        conf['batch_run_id'] = ''

    t = calculate_t(conf[FIELD_DATASET_NAME])
    r = calculate_lower_bound(t, 1)
    conf['r'] = r

    return conf


def bne_tune():
    global global_algorithm
    space = space_map[global_algorithm]

    conf = get_bne_config()
    logger.info(f'get runtime config: {conf}')
    modify_config(conf, space)
    logger.info(f'modify config finish')

    return bne(conf)


def random_select_param(space):
    a = random.choice(space.pos_args)
    return a.obj


def run_diff_dataset_bne_result(run_dataset_list=None):
    """
    运行不同数据集BNE算法的结果
    """
    global global_algorithm, global_dataset_name
    result = {}

    run_dataset_list = settle_dataset(run_dataset_list)
    logger.info(f'使用的数据集为 {run_dataset_list}')
    global_algorithm = Classify_Decision_Tree

    space = space_map[global_algorithm]
    params = {}
    for k, v in space.items():
        params[k] = random_select_param(v)

    for dataset_name in run_dataset_list:
        logger.info(f'{dataset_name} 数据集开始执行BNE算法')
        global_dataset_name = dataset_name

        conf = get_bne_config()
        logger.info(f'get runtime config: {conf}')
        modify_config(conf, params)
        logger.info(f'modify config finish')

        result[dataset_name] = bne(conf)

    return result


def algorithm_iter_classify():
    global global_algorithm
    result = {}
    for k in classifiers_map.keys():
        if k != Classify_Nearest_Neighbors:
            continue
        global_algorithm = k
        result[k] = bne_tune()
    return result


def algorithm_iter_clustering():
    global global_algorithm
    result = {}
    for k in default_clustering_conf.keys():
        global_algorithm = k
        result[k] = bne_tune()
    return result


def dataset_iter():
    """
    对多个数据集运行BNE算法
    """
    return run_diff_dataset_bne_result("all")


def modify_config(conf: dict, param: dict):
    """
    修改运行时配置，将超参数调优参数增加到运行时配置中。

    :param conf: 运行时配置文件，bne算法的配置
    :param param: tune超参数调优的调优参数
    """
    logger.info(f'conf: {conf}, param: {param}')

    keys = []
    for k, v in param.items():
        conf[k] = param[k]
        keys.append(k)

    # 设置不执行总数据集
    set_dict_with_log(conf, 'population_run', False)

    # 增加运行时配置的
    set_dict_with_log(conf, 'runtime_key', keys)

    # 获取执行model类别
    set_dict_with_log(conf, FIELD_MODEL_TYPE, get_model_type(conf[FIELD_ALGORITHM]))

    # 增加超时停止迭代
    set_dict_with_log(conf, 'skip_iter_time', 300.0)

    # 增加分数变化停止迭代
    set_dict_with_log(conf, 'stop_iter_score', {'percent': 0.001})


# 数据集执行的全局变量，todo 尽量不要用全局变量
global_dataset_name = ds_const.Dataset_name_SUSY
global_algorithm = ALGORITHM_kmeans
batch_run_id = None

if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间
    res = algorithm_iter_classify()
    end_time = time.time()  # 记录结束时间
    run_time = end_time - start_time
    print("代码运行时间为：{}秒".format(run_time))
    print(res)
