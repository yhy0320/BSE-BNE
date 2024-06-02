import __init__
from model.result import *
from dataset_util.common import split_data_and_label, load_conf


def load_runtime_conf_by_run_id(run_id: str):
    """
    根据 run_id 读取当时的 runtime 配置
    """
    df = query({
        'run_id': run_id,
    })
    file_path = df.iloc[0]['file_path']
    conf = load_conf(file_path)
    return conf

