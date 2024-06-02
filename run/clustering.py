import json
import time
import warnings
import numpy as np
import utils.utils_dict

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from run.metrics import metric_result
from run.field import *
from utils.utils_file import get_base_path
from utils.utils_log import logger
from run.model.result_kmeans import insert

algorithm_special_keys = [
    FIELD_K,
    'connectivity',
    'bandwidth'
]

with open(get_base_path() + "run/clustering.json", "r") as f:
    default_clustering_conf = json.load(f)


def construct_algorithm(params: dict):
    """
    根据配置文件装载算法模型
    """
    algo = params[FIELD_ALGORITHM]
    logger.info(f'run algo = {algo}')

    if algo == ALGORITHM_kmeans:
        return KMeans(
            n_clusters=params[FIELD_K],
            random_state=params['seed'],
        )

    if algo == ALGORITHM_gaussian_mixture:
        return GaussianMixture(
            n_components=params['k'],
            covariance_type="full",
            random_state=params['seed'],
        )

    if algo == ALGORITHM_mean_shift:
        set_default_params(params, ALGORITHM_mean_shift, ['bandwidth'])
        return MeanShift(
            bandwidth=params['bandwidth'],
            bin_seeding=True,
        )

    if algo == ALGORITHM_agglomerative_ward_clustering:
        set_default_params(params, ALGORITHM_agglomerative_ward_clustering, ['connectivity'])
        return AgglomerativeClustering(
            n_clusters=params["k"],
            linkage="ward",
            connectivity=params['connectivity'],
        )

    if algo == ALGORITHM_agglomerative_average_clustering:
        set_default_params(params, ALGORITHM_agglomerative_average_clustering, ['connectivity'])
        return AgglomerativeClustering(
            linkage="average",
            n_clusters=params["k"],
            connectivity=params['connectivity'],
        )

    if algo == ALGORITHM_spectral_clustering:
        return SpectralClustering(
            n_clusters=params["k"],
            eigen_solver="arpack",
            affinity="nearest_neighbors",
            random_state=params['seed'],
        )

    if algo == ALGORITHM_dbscan:
        set_default_params(params, ALGORITHM_dbscan, ['eps'])
        return DBSCAN(
            eps=params["eps"],
        )

    if algo == ALGORITHM_optics:
        set_default_params(params, ALGORITHM_optics, ['min_samples', 'xi', 'min_cluster_size'])
        return OPTICS(
            min_samples=params["min_samples"],
            xi=params["xi"],
            min_cluster_size=params["min_cluster_size"],
        )

    if algo == ALGORITHM_affinity_propagation:
        set_default_params(params, ALGORITHM_affinity_propagation, ['damping', 'preference'])
        return AffinityPropagation(
            damping=params["damping"],
            preference=params["preference"],
            random_state=params['seed'],
        )

    if algo == ALGORITHM_birch:
        return Birch(
            n_clusters=params["k"],
        )

    assert False, f"啥算法啊你要用, algorithm = {algo}"


def run_clustering(ds, l, params: dict):
    """
    运行聚类算法，返回分数和消耗时间
    """
    logger.info(f'运行分类算法，算法为 {params["algorithm"]}')

    algorithm = construct_algorithm(params)
    logger.info(f'get model: {algorithm}')

    X = ds.to_numpy(dtype=float)
    # 如果使用的是谱聚类或BIRCH算法，则将X转换为连续的内存块，以提高算法的性能
    if algorithm == ALGORITHM_spectral_clustering or algorithm == ALGORITHM_birch:
        X = np.ascontiguousarray(X)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        start = time.time()
        algorithm.fit(X)
        cost_time = time.time() - start

    logger.info(f'finish model fit, cost_time = {cost_time}')

    if hasattr(algorithm, "labels_"):
        y_pred = algorithm.labels_.astype(int)
    else:
        y_pred = algorithm.predict(X)

    logger.info(f'finish predict result')

    metrics = metric_result(l, y_pred)
    logger.info(f'finish predict metrics = {metrics}')

    if params["algorithm"] == ALGORITHM_kmeans:
        conf = params.copy()
        conf = utils.utils_dict.merge_dict(conf, metrics)
        conf['instance'] = ds.shape[0]

        insert(conf)

    return metrics, cost_time


def set_default_params(src: dict, algo: str, args: list):
    """
    更新配置，增加默认值
    """
    for arg in args:
        if arg not in src:
            logger.info(f'set default value, {arg} = {default_clustering_conf[algo][arg]}')
            src[arg] = default_clustering_conf[algo][arg]


def get_clustering_list() -> list:
    """
    返回当前可用的聚类算法列表
    """
    algorithms = []
    for k in default_clustering_conf.keys():
        algorithms.append(k)

    return algorithms
