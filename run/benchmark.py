"""
直接运行聚类算法，得到基准执行结果
"""

import json
import time
import uuid

import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_util.common import get_local_exist_dataset_list
from dataset_util.dataset_name_consts import *
from dataset_util.utils_csv import split_data_and_label
from dataset_util.utils_rsp import D, DS_TYPE_D, load_d_config
from run.bne import insert_bne_result
from run.clustering import run_clustering
from run.model.result_kmeans import query
from utils.utils_dict import order_dumps, extract_special_config
from utils.utils_file import get_output_dirname
from utils.utils_log import logger

batch_run_id = ''
global_dataset_name = Dataset_name_HIGGS


def RUN_benchmark(ds, label, conf):

    metrics, cost_time = run_clustering(ds, label, conf)

    new_conf = conf.copy()

    new_conf['instance'] = ds.shape[0]
    new_conf['dimension'] = ds.shape[1]
    new_conf['cost_time'] = cost_time
    new_conf['params'] = order_dumps(extract_special_config(conf, ['k']))
    new_conf['metrics'] = json.dumps(metrics)
    new_conf['file_path'] = ''
    new_conf['batch_run_id'] = conf['batch_run_id']
    new_conf['config'] = json.dumps(conf)

    logger.info(f"insert runtime result into mysql, result = {new_conf}")
    insert_bne_result(new_conf)


def gen_conf() -> dict:
    global batch_run_id

    conf = {
        'algorithm': 'kmeans',
        'k': 12,
        'seed': int(time.time()),
        'batch_run_id': batch_run_id,
        'dataset_name': Dataset_name_Covtype,
        'run_id': uuid.uuid1().__str__(),
    }

    return conf


def draw():

    clustering_score = [
        'homogeneity_score',
        'completeness_score',
        'v_measure_score',
        'adjusted_rand_score',
        'adjusted_mutual_info_score',
        'fowlkes_mallows_score',
        'rand_score',
        'mutual_info_score',
        'normalized_mutual_info_score',
    ]

    dataset_name = Dataset_name_Covtype

    group = query({
        'dataset_name': dataset_name,
    }).groupby('instance')

    for score in clustering_score:
        plt.clf()
        x = []
        y = []
        for instance, df in group:
            x.append(int(instance))
            m = df[[score]].mean(0)
            y.append(float(m[score]))

        dir_name = get_output_dirname() + "benchmark/"
        plt.plot(x, y)
        plt.savefig(dir_name + f"{dataset_name}_{score}.png")


def run():
    global batch_run_id, global_dataset_name
    batch_run_id = uuid.uuid1().__str__()

    conf = load_d_config(global_dataset_name)

    d = D(global_dataset_name, DS_TYPE_D)
    pbar = tqdm(range(10, min(1010, conf['instance']), 10))

    for ti in pbar:
        for i in range(5):
            pbar.set_description(f'[num={ti}][i={i}]')
            df = d.sample_dataframe(ti)
            ds, label = split_data_and_label(df)
            RUN_benchmark(ds, label, gen_conf())


def run_draw():
    draw()


if __name__ == '__main__':
    for _ in range(50):
        for dataset_name in get_local_exist_dataset_list():
            if dataset_name == Dataset_name_PA or \
                dataset_name == Dataset_name_covid or \
                dataset_name == Dataset_name_Covtype or \
                dataset_name == Dataset_name_skin or \
                dataset_name == Dataset_name_Dry_Bean or \
                    dataset_name == Dataset_name_SUSY or \
                    dataset_name == Dataset_name_circle or \
                    dataset_name == Dataset_name_mnist:
                continue
            print(f'>>> {dataset_name}')
            global_dataset_name = dataset_name
            run()
