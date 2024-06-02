from run.bne_env import load_runtime_conf_by_run_id
import run.model.global_result as gr
from run.bne_ana import draw_sample_and_population_result


def run_global_result(run_id: str):
    """
    运行总体数据集
    """
    print(f'执行总数据集 run_id = {run_id}')
    conf = load_runtime_conf_by_run_id(run_id)
    print(f'获取到运行时配置 = {conf}')
    gr.run_global_result(conf, True)
    draw_sample_and_population_result(conf['run_id'])