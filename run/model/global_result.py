"""
这个文件用于储存总体数据集的运算结果，所需的字段为：

dataset_name:str    数据集名称
algorithm:str       算法名称
function_name:str   插入结果的函数名
cost_time:int       算法执行时间
timestamp:int       执行的时间戳
*args:Any           任何的c
"""

import json
import time
import numpy as np
import pandas as pd
from dataset_util.utils_csv import load_csv_data_and_label, load_csv, save_csv
from dataset_util.utils_mysql import get_target_dataset_result_from_mysql_v2, save_to_mysql

from utils.utils_log import logger
from utils.utils_dict import order_dumps, extract_special_config
from run.clustering import run_clustering, algorithm_special_keys
from run.classifier import run_classify
from dataset_util.utils_mysql import get_mysql_engine
from run.consts import *
from run.field import *
from dataset_util.common import split_data_and_label
from dataset_util.utils_csv import load_csv_test

# 全局运行结果文件名称
GLOBAL_RESULT_FILE_NAME = "global_result.csv"


# 插入数据所需的keys
global_result_special_keys = [
    'dataset_name',
    'instance',
    'dimension',
    'cost_time',
    'params',
    'metrics',
    'file_path',
    'run_id',
]


# 初始化全局运行文件
def get_global_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            'dataset_name',
            'algorithm',
            'function_name',
            'cost_time',
            'timestamp',
        ],
    )


# mind: 插入运行结果
def run_global_result(conf: dict, rerun_flag=False) -> (dict, float):
    """
    运行总数据集聚类算法，如果已经运行过，直接返回结果
    """
    logger.info(f'运行总体数据集, runtime conf = {conf}')

    # 检查是否运行过
    params = algorithm_special_keys.copy()
    params.append('dataset_name')

    df = check_if_exist_global_result(extract_special_config(conf, params))
    if not rerun_flag and df is not None:
        logger.info(f'get exist result: {df}')
        return json.loads(df.loc[0, 'metrics']), df.loc[0, 'cost_time']

    logger.info(f'load dataset: START')
    ds, l = load_csv_data_and_label(conf['dataset_name'])
    logger.info(f'load dataset: FINISH')

    logger.info(f'运行配置 params = {extract_special_config(conf, conf["runtime_key"])}, seed = {conf["seed"]}')
    metrics = {}
    cost_time = 0.0

    if conf[FIELD_MODEL_TYPE] == CONSTS_MODEL_TYPE_CLUSTERING:
        metrics, cost_time = run_clustering(ds, l, conf)
    elif conf[FIELD_MODEL_TYPE] == CONSTS_MODEL_TYPE_CLASSIFY:
        metrics, cost_time = run_classify(ds, l, conf)

    logger.info(f'执行结果 metrics = {metrics}')

    insert_bne_global_result({
        'dataset_name': conf['dataset_name'],
        'instance': ds.shape[0],
        'dimension': ds.shape[1],
        'cost_time': cost_time,
        'params': order_dumps(extract_special_config(conf, conf["runtime_key"])),
        'file_path': conf['dir_name'],
        'metrics': json.dumps(metrics),
        'run_id': conf['run_id'],
    })

    logger.info(f'save global result to mysql')

    return metrics, cost_time


def check_if_exist_global_result(conf: dict):
    """
    检查 global_result 是否有历史数据
    """
    logger.info(f'conf: {conf}')
    df = get_target_dataset_result_from_mysql_v2(conf['dataset_name'], 'bse_bne')
    if df is None or df.shape[0] == 0:
        logger.info(f'没有发现历史数据')
        return None

    params = order_dumps(conf)
    logger.info(f'params {params}')

    df = df.query('params == @params')
    if df.shape[0] == 0:
        return None

    return df.loc[:0]


def construct_result_dataframe():
    """
    global_result 结构体
    """
    df = pd.DataFrame(columns=[
        'dataset_name',
        'instance',
        'dimension',
        'cost_time',
        'params',
        'metrics',
        'file_path',
        'run_id',
    ])

    return df


def construct_dataframe_row(conf: dict):
    """
    构造 mysql 插入数据的结构
    """
    df = construct_result_dataframe()
    row = extract_special_config(conf, global_result_special_keys)
    df = df.append(row, ignore_index=True)
    return df


def insert_bne_global_result(conf: dict):
    """
    插入一条新的数据
    """
    engine = get_mysql_engine('bse_bne')
    df = construct_dataframe_row(conf)
    df.to_sql('all_data_result', engine, if_exists='append', index=False)


def query(conf: dict):
    """
    查询
    """
    engine = get_mysql_engine('bse_bne')

    sql = "select * from all_data_result"

    if len(conf) > 0:
        sql += " where "
        conditions = []
        for k, v in conf.items():
            conditions.append(f"{k} = '{v}'")

        sql += ' AND '.join(conditions)

    sql += " order by create_time desc"

    df = pd.read_sql(sql, engine)
    # print(df)
    return df
