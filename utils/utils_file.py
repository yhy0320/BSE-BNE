import os

from .utils_time import *
from .utils_topic import gen_failed_sub_topic, drop_failed_sub_topic

result_dir_name = "result/"
output_dir_name = "output/"


def get_base_path() -> str:
    """
    获取整个项目目录的基地址
    """
    path = str(__file__)
    idx = path.find("BSE-BNE")

    return path[:idx + 8]


def get_result_dir_path():
    """
    生成当天的 result 文件夹路径
    <base_path>/<result>/<date>/
    """
    base_path = get_base_path()
    now = get_cur_date()

    dir_name = base_path + result_dir_name + now + "/"
    os.makedirs(dir_name, exist_ok=True)

    return dir_name


def get_topic_result_dir_path(topic: str, has_failed_prefix=False) -> str:
    """
    生成某个 topic 下的 result 结果保存路径
    :param topic: 主题
    :param has_failed_prefix: 是否需要增加失败任务前缀
    """
    dir_name = get_result_dir_path() + topic + "/"
    if has_failed_prefix:
        dir_name += gen_failed_sub_topic(get_cur_time()) + "/"
    else:
        dir_name += get_cur_time() + "/"
    os.makedirs(dir_name, exist_ok=True)

    return dir_name


def get_topic_result_dir_by_time(topic: str, date: str, time: str) -> str:
    """
    生成当天当时的 result 文件夹路径
    <base_path>/<result>/<date>/<timestamp>/
    """
    base_path = get_base_path()
    dir_name = base_path + result_dir_name + date + "/" + topic + "/" + time + "/"
    assert os.path.exists(dir_name), f"不存在该实验结果, dir_name = {dir_name}"

    return dir_name


def get_last_topic_result_dir(topic: str) -> str:
    dir_name = get_base_path() + result_dir_name
    date_list = os.listdir(dir_name)
    date_list.sort(key=lambda x: x, reverse=True)
    is_find = False
    for date in date_list:
        topic_list = os.listdir(dir_name + date + "/")
        if topic in topic_list:
            dir_name = dir_name + date + "/" + topic + "/"
            is_find = True
            break

    if not is_find:
        assert False, "好像没这个topic噢！"

    time_list = os.listdir(dir_name)
    time_list.sort(key=lambda x: x, reverse=True)
    assert len(time_list) > 0, "这个topic好像没有数据吧？？？"
    return dir_name + time_list[0] + "/"


def extract_dir_name(file_path: str) -> str:

    idx = file_path.rfind('/')
    dir_name = file_path[:idx + 1]
    return dir_name


def __pass_():
    pass


def mark_failed_prefix(dir_name: str):
    """
    标记为不成功执行结果
    """
    # /home/biang/Desktop/data_split/result/2023-01-03/_bne_v2_10000_20_2_[-1000,1000]_dataset/12:01:24
    dir_list = dir_name.split("/")
    assert len(dir_list) >= 2, "这啥文件路径啊"

    if dir_list[-1] != "":
        dir_list[-1] = gen_failed_sub_topic(dir_list[-1])
    elif dir_list[-2] != "":
        dir_list[-2] = gen_failed_sub_topic(dir_list[-2])

    dir_name = "/".join(dir_list)

    os.rename(dir_name, gen_failed_sub_topic(dir_name))

    return dir_name


def drop_failed_prefix(dir_name: str):
    """
    标记为成功执行结果
    """
    # /home/biang/Desktop/data_split/result/2023-01-03/_bne_v2_10000_20_2_[-1000,1000]_dataset/12:01:24

    new_dir_name = drop_dir_name_failed_prefix(dir_name)
    os.rename(dir_name, new_dir_name)

    return dir_name


def drop_dir_name_failed_prefix(dir_name: str) -> str:
    """
    删除路径名最后文件的 failed 前缀
    """
    dir_list = dir_name.split("/")
    assert len(dir_list) >= 2, "这啥文件路径啊"

    if dir_list[-1] != "":
        dir_list[-1] = drop_failed_sub_topic(dir_list[-1])
    elif dir_list[-2] != "":
        dir_list[-2] = drop_failed_sub_topic(dir_list[-2])

    new_dir_name = "/".join(dir_list)

    return new_dir_name


def get_output_dirname() -> str:
    """
    获取 output 文件夹位置
    """
    base_path = get_base_path()
    return base_path + output_dir_name
