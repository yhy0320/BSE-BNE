"""
这个地方是BNE算法分析用代码
"""
import json
import math
import os
import shutil
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score
from tqdm import tqdm

from dataset_util.common import split_data_and_label, load_conf, load_dataset_conf
from dataset_util.utils_csv import save_csv, load_csv
from dataset_util.utils_draw import square_figs
from dataset_util.utils_rsp import select_rsp_list, load_rsp_config
from model import thesis
from run.algorithm_model import get_model_type
from run.bne_config import load_bne_config
from run.consts import *
from run.field import *
from run.metrics import clustering_metrics
from run.model.global_result import query as global_query
from run.model.result import cancel_favorite as model_cancel_favorite
from run.model.result import favorite as model_favorite
from run.model.result import query, update
from utils.utils_dict import list_to_map
from utils.utils_file import get_base_path, get_last_topic_result_dir, get_output_dirname
from utils.utils_log import logger
from utils.utils_time import get_cur_timestamp

bne_different_dataset_result_dir_name = "bne_ddr/"


def get_multi_run_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            FIELD_DATASET_NAME,
            FIELD_BLOCK_NUM,
            FIELD_BLOCK_SIZE,
            FIELD_RATIO,
            FIELD_INSTANCE,
            FIELD_K,
            FIELD_SCORE,
            FIELD_P_SCORE,
        ]
    )


def get_ana_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            FIELD_DATASET_NAME,
            FIELD_BLOCK_NUM,
            FIELD_BLOCK_SIZE,
            FIELD_RATIO,
            FIELD_INSTANCE,
            FIELD_K,
            FIELD_SCORE,
            FIELD_P_SCORE,
            FIELD_MEAN,
            FIELD_VAR,
        ]
    )


def parse_bne_result(col_name: str, topic: str, x_col_name: str, y_col_names: list):
    """
    解释BNE算法的产物，根据 score 是否在 p_score 的范围之内，决定RSP块的数量。
    确定了 RSP 块的数量之后，随机获取RSP块并执行多次取平均值。

    该函数为【数据生成函数】

    数据结构变化: result.csv -> multi_run.csv

    :param col_name: 分组key
    :param topic: 主题
    :param x_col_name: x轴的名字
    :param y_col_names: y轴的名字列表

    """
    logger.info(f'col_name: {col_name}, topic: {topic}, x_col_name: {x_col_name}, y_col_names: {y_col_names}')

    # 获取任务结果文件夹
    dir_name = get_last_topic_result_dir(topic)
    print(f'Load {dir_name + RESULT_CSV_FILE_NAME}')
    df = load_csv(dir_name + RESULT_CSV_FILE_NAME)
    logger.info(f'load result shape: {df.shape}')
    group = df.groupby(col_name)
    # 获取实验所使用的keys
    keys = list(group.indices.keys())

    # 获取运行时配置
    run_conf = load_conf(dir_name)
    logger.info(f'load runtime config: {run_conf}')
    # 获取原数据集名字
    dataset_name = run_conf[FIELD_DATASET_NAME]

    # 加载RSP数据块配置文件
    rsp_conf = load_rsp_config(dataset_name)
    logger.info(f'load rsp config: {rsp_conf}')

    result = get_multi_run_result()

    for k in keys:

        group_df = df.iloc[group.indices[k]].copy()

        p_score = group_df.loc[group_df.index[0]][FIELD_P_SCORE]
        p_cost_time = group_df.loc[group_df.index[0]][FIELD_P_COST_TIME]
        logger.info(f'get cur "k" kmeans score: {p_score}')

        upper = p_score * (1 + 0.005)  # 范围
        lower = p_score * (1 - 0.005)
        match = group_df[(group_df[FIELD_SCORE] >= lower) & (group_df[FIELD_SCORE] < upper)]
        satisfied_num = -1
        if match.shape[0] > 0:
            satisfied_num = match.iloc[0][FIELD_BLOCK_NUM]
        logger.info(f'get satisfied num: {satisfied_num}')

        pbar = tqdm(range(20))

        for ti in pbar:
            pbar.set_description(f'k = {k}')

            logger.info(f'start load rsp block, num = {satisfied_num}')
            rsp_list, rsp_block_total_num = select_rsp_list(dataset_name, satisfied_num)
            assert len(rsp_list) > 0, "这啥啊"
            assert len(rsp_list) == satisfied_num or len(rsp_list) == rsp_block_total_num, "这数字好像不满足"
            logger.info(f'load rsp block, num = {len(rsp_list)}')

            sample = pd.concat(rsp_list, axis=0, ignore_index=True)
            ds, l = split_data_and_label(sample)
            logger.info(f'load sample, shape = {sample.shape}')

            logger.info(f'sample run kmeans: START')
            start = time.time()
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(ds.to_numpy(dtype=float))
            cost_time = time.time() - start
            logger.info(f'sample run kmeans: FINISH')
            score = fowlkes_mallows_score(l, kmeans.labels_)
            logger.info(f'sample kmeans score: {score}')

            result = result.append({
                FIELD_DATASET_NAME: dataset_name,
                FIELD_BLOCK_NUM: satisfied_num,
                FIELD_BLOCK_SIZE: rsp_conf[FIELD_BLOCK_SIZE],
                FIELD_INSTANCE: rsp_conf[FIELD_INSTANCE],
                FIELD_RATIO: sample.shape[0] / rsp_conf[FIELD_INSTANCE],
                FIELD_K: k,
                FIELD_SCORE: score,
                FIELD_COST_TIME: cost_time,
                FIELD_P_SCORE: p_score,
                FIELD_P_COST_TIME: p_cost_time,
            }, ignore_index=True)
            logger.info(f'insert new record, in {ti} time')

    save_csv(dir_name + MULTI_RUN_CSV_FILE_NAME, result)
    logger.info(f'save {MULTI_RUN_FILE_NAME} file in {dir_name}')


def draw_bne_algorithm_result(col_name: str, topic: str, x_col_name: str, y_col_names: list):
    """
    这个函数主要是用来画BNE算法产生的数据图。
    该图主要是比较不同RSP块数据集的运算结果与总体数据集结果的变化过程。
    需要关注的有两条线，一条是score，一条是p_score。

    该函数为【图生成函数】
    """
    dir_name = get_last_topic_result_dir(topic)
    print(f'Load {dir_name + RESULT_CSV_FILE_NAME}')
    df = load_csv(dir_name + RESULT_CSV_FILE_NAME)
    group = df.groupby(col_name)

    fig = plt.figure()
    keys = list(group.indices.keys())

    for idx, ax in enumerate(square_figs(fig, group.ngroups)):
        group_df = df.iloc[group.indices[keys[idx]]].copy()
        # 画指标变化曲线
        for y_col_name in y_col_names:
            ax.plot(group_df[x_col_name], group_df[y_col_name])

        p_score = group_df.loc[group_df.index[0]]['p_score']  # 全局计算存储
        # 计算满足范围的边界
        upper = p_score * (1 + 0.005)
        lower = p_score * (1 - 0.005)
        satisfied_num = group_df[(group_df['score'] >= lower) & (group_df['score'] < upper)].iloc[0]['block_num']
        # 画图
        ax.axvline(satisfied_num, linestyle='--', c='grey')
        ax.fill_between(group_df[x_col_name], upper, lower, alpha=0.1, color='red')

    fig.tight_layout()
    fig.savefig(dir_name + DRAW_PLOT_PNG_FILE_NAME)
    print(f'Save figure in {dir_name + DRAW_PLOT_PNG_FILE_NAME}')


def transfer_multi_run_2_static_res(col_name: str, topic: str):
    """
    这个函数就把 multi_run 的结果转成论文使用的 static_run

    multi_run.csv -> static_run
    """
    dir_name = get_last_topic_result_dir(topic)
    print(f'Load {dir_name + MULTI_RUN_CSV_FILE_NAME}')
    df = load_csv(dir_name + MULTI_RUN_CSV_FILE_NAME)
    group = df.groupby(col_name)
    #
    keys = list(group.indices.keys())

    result = get_ana_result()

    for k in keys:
        group_df = df.iloc[group.indices[k]].copy()
        group_df['diff'] = group_df['score'] - group_df['p_score']

        idx = group_df.index[0]

        result = result.append({
            FIELD_DATASET_NAME: group_df.loc[idx][FIELD_DATASET_NAME],
            FIELD_BLOCK_NUM: group_df.loc[idx][FIELD_BLOCK_NUM],
            FIELD_BLOCK_SIZE: group_df.loc[idx][FIELD_BLOCK_SIZE],
            FIELD_RATIO: group_df.loc[idx][FIELD_RATIO],
            FIELD_INSTANCE: group_df.loc[idx][FIELD_INSTANCE],
            FIELD_K: int(k),
            FIELD_SCORE: group_df[FIELD_SCORE].mean(),
            FIELD_COST_TIME: group_df[FIELD_COST_TIME].mean(),
            FIELD_P_SCORE: group_df[FIELD_P_SCORE].mean(),
            FIELD_P_COST_TIME: group_df[FIELD_P_COST_TIME].mean(),
            FIELD_MEAN: group_df[FIELD_DIFF].mean(),
            FIELD_VAR: group_df[FIELD_DIFF].var(),
        }, ignore_index=True)

    save_csv(dir_name + STATIC_RESULT_CSV_FILE_NAME, result)


def extract_metrics(x):
    metrics = json.loads(x['metrics'])
    return metrics['fowlkes_mallows_score']


def extract_params(x):
    parms = list_to_map(json.loads(x['params']))
    return parms['k']


def ana_score(score) -> float:
    l = score.tolist()
    n = len(l)
    diff = 0
    for i in range(1, n):
        diff += score[i] - score[i - 1]
    # print(l)
    # print(diff)

    return diff


def draw_from_mysql(conditions: dict):
    """
    从 mysql 数据库读取数据画图
    """
    df = query(conditions)
    df = df.sort_values(by='instance')
    if df.shape[0] <= 0:
        logger.error(f'error condition = {conditions}')
        return

    params = df.iloc[0]['params']  # 获取执行参数
    algorithm = list_to_map(json.loads(params))["algorithm"]
    model_type = get_model_type(algorithm)

    score_keys = []
    if model_type == CONSTS_MODEL_TYPE_CLUSTERING:
        df, score_keys = extract_all_score(df)
    elif model_type == CONSTS_MODEL_TYPE_CLASSIFY:
        df, score_keys = extract_classify_score(df)

    df['cost_time'] = df['cost_time'].apply(pd.to_numeric) * 1000
    df['cost_time'] = df['cost_time'].round(4)

    for key in score_keys:
        plt.clf()
        # plt.plot(df['instance'], df[key], label=key)
        # plt.plot(df['instance'], df['cost_time'], label='cost_time')
        # plt.title(f'{params_format(params)}\n {algorithm} {conditions["run_id"]}', fontsize=10)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(get_base_path() + "pics/" + f"{conditions['run_id']}_{key}.png")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(df['instance'], df[key], label=key, marker='o')
        ax.set_ylabel("score")

        ax1 = ax.twinx()
        ax1.plot(df['instance'], df['cost_time'], label="cost", c='y')
        ax1.set_ylabel("cost time")

        plt.title(f'{params_format(params)}\n {algorithm} {conditions["run_id"]}', fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(get_base_path() + "pics/" + f"{conditions['run_id']}_{key}_cost_time.png")
        print(get_base_path() + "pics/" + f"{conditions['run_id']}_{key}_cost_time.png")

    return df[score_keys]


def params_format(params) -> str:
    ret = ""

    if type(params) == str:
        params_list = json.loads(params)
        for param in params_list:
            if param[0] == "algorithm":
                continue
            if type(param[1]) == float:
                param[1] = round(param[1], 4)
            ret = ret + f"[{param[0]}: {param[1]}]"
        return ret

    if type(params) == dict:
        for k, v in params.items():
            if k == "algorithm":
                continue
            if type(v) == float:
                v = round(v, 4)
            ret = ret + f"[{k}: {v}]"
        return ret

    if type(params) == list:
        params_list = params
        for param in params_list:
            if param[0] == "algorithm":
                continue
            if type(param[1]) == float:
                param[1] = round(param[1], 4)
            ret = ret + f"[{param[0]}: {param[1]}]"
        return ret

    return ""


def get_run_id_by_dir_name(dir_name: str) -> str:
    # /home/biang/Desktop/data_split/result/2023-01-04/_bne_10000_20_2_dataset/19:42:55
    # ed9a03c0-8c24-11ed-a1de-25f45858117a
    conf = load_conf(dir_name)
    # print(conf)
    if 'run_id' not in conf:
        return "-"
    return conf['run_id']


def extract_run_id(file_path: str) -> str:
    run_id = file_path.split('/')[-1].split('.')[0]
    l = run_id.split('_')
    if len(l) > 1:
        run_id = l[0]

    return run_id


def check_local_favorite(pics_name: str) -> bool:
    dir_name = get_base_path() + "star_pics/"
    return os.path.exists(dir_name + pics_name)


def __check_tag(run_id: str):
    df = query({
        'run_id': run_id,
    })

    if df.shape[0] <= 0:
        logger.error(f'不对啊，咋没这个数据啊 run_id = {run_id}')
        print('不对啊，咋没这个数据啊')
        return 'error'

    tag = df.iloc[0]['tag']

    return tag


def check_tag(file_path: str):
    run_id = extract_run_id(file_path)
    return __check_tag(run_id)


def check_favorite(file_path: str):
    """
    检查该文件是需要加入收藏还是取消收藏
    """

    tag = check_tag(file_path)

    if tag == 'star':  # 已经在收藏了
        cancel_favorite(file_path)
        return 'cancel favorite'
    else:
        favorite(file_path)
        return 'favorite'


def favorite(file_path: str):
    """
    收藏
    """
    dst = file_path.replace('pics', 'star_pics')
    if not os.path.exists(dst):
        shutil.copyfile(file_path, dst)
        print(f'copy file: {file_path} -> {dst}')
        logger.info(f'copy file: {file_path} -> {dst}')

    run_id = extract_run_id(file_path)
    print(f'收藏 run_id = {run_id}')
    logger.info(f'收藏 run_id = {run_id}')
    model_favorite(run_id)


def cancel_favorite(file_path: str):
    """
    取消收藏
    """
    dst = file_path.replace('pics', 'star_pics')

    if os.path.exists(dst):
        os.remove(dst)
        print(f'delete file: {dst}')
        logger.info(f'delete file: {dst}')

    run_id = extract_run_id(file_path)
    print(f'取消收藏 run_id = {run_id}')
    logger.info(f'取消收藏 run_id = {run_id}')
    model_cancel_favorite(run_id)


def archive():
    """
    压缩视频文件
    时间格式: "%Y-%m-%d %H:%M:%S"
    """
    file_map = {}

    # 图片路径
    dir_name = get_base_path() + "pics/"
    # 获取文件
    pics_list = os.listdir(dir_name)

    # 当天不压缩
    cur_date = time.strftime("%Y-%m-%d", time.localtime())

    # 收集文件列表
    for file_path in pics_list:
        if file_path.endswith('.zip'):
            continue
        t = time.strftime("%Y-%m-%d", time.localtime(os.path.getctime(dir_name + file_path)))
        if t >= cur_date:
            continue
        if t not in file_map:
            file_map[t] = [file_path]
        else:
            file_map[t].append(file_path)

    # 开始压缩
    for k, file_list in file_map.items():
        # 创建当天文件名路径
        z = zipfile.ZipFile(dir_name + f'{k}.zip', 'w', zipfile.ZIP_DEFLATED)
        for file in file_list:
            z.write(dir_name + file, file)
        z.close()

        # 删除文件
        for file in file_list:
            os.remove(dir_name + file)


def extract_classify_score(df: pd.DataFrame) -> (pd.DataFrame, list):
    """
    获取分类算法指标数据
    """
    keys = ['score']

    global_score_key = ""

    def score_key(x):
        metrics = json.loads(x['metrics'])
        return metrics[global_score_key]

    for key in keys:
        global_score_key = key
        df[global_score_key] = df.apply(score_key, axis=1)

    return df, keys


def extract_all_score(df: pd.DataFrame) -> (pd.DataFrame, list):
    """
    获取聚类算法指标数据
    """
    keys = [k for k in clustering_metrics.keys()]

    global_score_key = ""

    def score_key(x):
        metrics = json.loads(x['metrics'])
        return metrics[global_score_key]

    for key in keys:
        global_score_key = key
        df[global_score_key] = df.apply(score_key, axis=1)

    return df, keys


def draw_sample_and_population_result(run_id: str):
    conditions = {
        'run_id': run_id,
    }

    df = query(conditions)
    df = df.sort_values(by='instance')

    params = df.iloc[0]['params']  # 获取执行参数
    algorithm = list_to_map(json.loads(params))["algorithm"]
    model_type = get_model_type(algorithm)

    p_df = global_query(conditions)
    if p_df.shape[0] <= 0:
        logger.info(f'没有总体数据集啊, run_id = {run_id}')
        return

    score_keys = []
    if model_type == CONSTS_MODEL_TYPE_CLUSTERING:
        df, score_keys = extract_all_score(df)
        p_df, _ = extract_all_score(p_df)
    elif model_type == CONSTS_MODEL_TYPE_CLASSIFY:
        df, score_keys = extract_classify_score(df)
        p_df, _ = extract_classify_score(p_df)

    for key in score_keys:
        plt.clf()
        plt.plot(df['instance'], df[key], label=key, marker='o', linestyle='-.')
        plt.axhline(y=p_df.iloc[0][key], label=f'global', c='y')
        plt.title(f'{params_format(params)}\n {algorithm} {conditions["run_id"]}', fontsize=10)
        plt.legend()
        plt.tight_layout()

        last_record = df.iloc[-1]

        thesis.insert_thesis({
            'params': params,
            'dataset_name': df.iloc[0]['dataset_name'],
            'sample': last_record['instance'],
            'instance': p_df.iloc[0]['instance'],
            'dimension': p_df.iloc[0]['dimension'],
            'cost_time': last_record['cost_time'],
            'p_cost_time': p_df.iloc[0]['cost_time'],
            'score': last_record[key],
            'p_score': p_df.iloc[0][key],
            'mean': '',
            'var': '',
            'file_path': '',
            'run_id': run_id,
            'tag': '',
            'ratio': last_record['instance'] / p_df.iloc[0]['instance'],
        })

        plt.savefig(get_output_dirname() + "global_result/" + f"{run_id}_{key}_global.png")

    return df[score_keys]


def wash_result_data():
    """
    洗旧数据的config文件到mysql数据库
    """

    df = query({})

    count = 0

    for idx in df.index:
        if df.iloc[idx]['config'] != "":
            continue
        config = json.dumps(load_bne_config(df.iloc[idx]['file_path']))
        if config == "{}" or config == "":
            continue
        index = int(df.iloc[idx]['id'])
        update(f"id={index}", f"config='{config}'")
        print(config)
        count += 1


def draw_diff_dataset_bne_result(run_diff_dataset_bne_result_global_variable: dict):
    """
    生成相同算法，不同数据集下，BNE得到的不同数据集的RSP块数估计
    """

    run_ids = []

    key_dimension = {}
    config_map = {}

    batch_run_id = ""

    for dataset_name, run_id in run_diff_dataset_bne_result_global_variable.items():
        run_ids.append(run_id)

        conditions = {
            'run_id': run_id,
        }

        df = query(conditions)
        df = df.sort_values(by='instance')

        params = df.iloc[0]['params']  # 获取执行参数
        algorithm = list_to_map(json.loads(params))["algorithm"]
        model_type = get_model_type(algorithm)

        if batch_run_id == "":
            batch_run_id = df.iloc[0]['batch_run_id']

        conf_dir_name = df.iloc[0]['file_path']
        conf = load_bne_config(conf_dir_name)

        score_keys = []
        if model_type == CONSTS_MODEL_TYPE_CLUSTERING:
            df, score_keys = extract_all_score(df)
        elif model_type == CONSTS_MODEL_TYPE_CLASSIFY:
            df, score_keys = extract_classify_score(df)

        for key in score_keys:
            if key not in key_dimension:
                key_dimension[key] = {
                    # dataset_name: [df['instance'], df[key]]
                    dataset_name: [[i for i in range(1, df.shape[0] + 1)], df[key]]
                }
                config_map[key] = {
                    dataset_name: conf
                }
            else:
                # key_dimension[key][dataset_name] = [df['instance'], df[key]]
                key_dimension[key][dataset_name] = [[i for i in range(1, df.shape[0] + 1)], df[key]]
                config_map[key][dataset_name] = conf

    for score, runtime_result in key_dimension.items():
        plt.figure(figsize=(12, 4.8))

        plt.clf()
        for dataset_name, x_y in runtime_result.items():
            plt.plot(x_y[0], x_y[1], label=dataset_name, marker='o')
            fp = x_y[0][-1]
            plt.axvline(fp, linestyle='--', alpha=0.6)
        plt.title(f'key = {score}', fontsize=10)
        plt.legend()
        plt.tight_layout()
        # 保存
        plt.savefig(
            get_output_dirname() + bne_different_dataset_result_dir_name + f"{get_cur_timestamp()}_{batch_run_id}.png")


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


def r(delta=0.9, t=1, tau=1.0, ds=1):
    _r = (2 * math.log(ds) * delta ** 2) / (t ** 2) * math.log(2 / tau)
    return _r


def re_ana_score(df: pd.DataFrame, stop_iter_score, _r=0) -> pd.DataFrame:
    if stop_iter_score is None:
        return df

    satisfied_time = 0
    last_score = math.inf
    score = math.inf
    cur_idx = -1

    success_time = 2
    if 'success_time' in stop_iter_score:
        success_time = stop_iter_score['success_time']

    for i in range(df.shape[0]):
        cur_idx = i
        last_score = score
        score = df.iloc[i]['score']

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

        if satisfied_time == success_time:
            break
        else:
            continue

    df = df[:cur_idx + 1]
    new_n = df.shape[0]

    if new_n <= 5:
        return df

    # i = random.randint(0, new_n - 2)

    return df[min(int(_r), new_n - 3):]


def draw_diff_dataset_by_batch_run_id(batch_run_id: str, re_conf=None) -> str:
    batch_df = query({
        'batch_run_id': batch_run_id,
    })

    key_dimension = {}
    config_map = {}

    group = batch_df.groupby('run_id')
    for i, df in group:
        df = df.sort_values(by='instance')
        dataset_name = df.iloc[0]['dataset_name']

        conf = load_dataset_conf(dataset_name)
        _r = r(ds=conf['cluster'] * conf['dim'])
        # continue

        params = df.iloc[0]['params']  # 获取执行参数
        algorithm = list_to_map(json.loads(params))["algorithm"]
        model_type = get_model_type(algorithm)

        conf = json.loads(df.iloc[0]['config'])

        score_keys = []
        if model_type == CONSTS_MODEL_TYPE_CLUSTERING:
            df, score_keys = extract_all_score(df)
        elif model_type == CONSTS_MODEL_TYPE_CLASSIFY:
            df, score_keys = extract_classify_score(df)

        df['index'] = np.arange(1, df.shape[0] + 1)
        df = re_ana_score(df, re_conf, _r=_r)

        if df.shape[0] <= 0:
            print(f'Skip {dataset_name}')
            continue

        for key in score_keys:
            if key not in key_dimension:
                key_dimension[key] = {
                    # dataset_name: [df['instance'], df[key]]
                    dataset_name: [df['index'], df[key]]
                }
                config_map[key] = {
                    dataset_name: conf
                }
            else:
                key_dimension[key][dataset_name] = [df['index'], df[key]]
                config_map[key][dataset_name] = conf

    # return
    file_list = []

    for score, runtime_result in key_dimension.items():
        # plt.figure(figsize=(12, 4.8))
        plt.figure(figsize=(6.4, 6))

        plt.clf()
        for dataset_name, x_y in runtime_result.items():
            plt.plot(x_y[0], x_y[1], label=dataset_name, marker='.')
            plt.axvline(x_y[0].iloc[x_y[0].shape[0] - 1], linestyle='--', alpha=0.6)
        plt.title(f'key = {score}', fontsize=10)
        plt.legend()
        plt.tight_layout()
        # 保存
        if re_conf is None:
            file_name = get_output_dirname() + bne_different_dataset_result_dir_name + f"{get_cur_timestamp()}_{batch_run_id}.png"
        else:
            file_name = get_output_dirname() + 'ddr_regen/' + f"{get_cur_timestamp()}_{batch_run_id}.png"

        plt.savefig(file_name)
        print("save: ", file_name)
        file_list.append(file_name)

    return file_list[0]


if __name__ == '__main__':
    draw_diff_dataset_by_batch_run_id('1e92b665-af90-11ed-8226-0567bf6f54bc', re_conf={
        'percent': 0.001,
    })
