from run.classifier import classifiers_map
from run.clustering import default_clustering_conf
from run.consts import *


def get_model_type(algorithm: str) -> str:
    """
    获取算法的模型类别
    """
    model_type = ""
    if algorithm in classifiers_map:
        model_type = CONSTS_MODEL_TYPE_CLASSIFY
    if algorithm in default_clustering_conf:
        model_type = CONSTS_MODEL_TYPE_CLUSTERING

    return model_type
