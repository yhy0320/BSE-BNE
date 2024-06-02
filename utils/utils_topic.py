"""
这个文件用来获取 topic 名称
"""


def gen_topic_by_func_and_dataset(func_name: str, dataset_name: str) -> str:
    """
    根据方法名和数据集名称获取到topic
    """

    return gen_topic_format(func_name) + gen_topic_format(dataset_name)


def gen_topic_format(sub_topic: str) -> str:
    """
    生成topic前缀
    """
    return f"_{sub_topic}"


def gen_failed_sub_topic(string: str) -> str:
    """
    生成失败任务前缀
    """
    return failed_sub_topic() + string


def failed_sub_topic() -> str:
    """
    失败任务前缀
    """
    return gen_topic_format('FAIL')


def drop_failed_sub_topic(string: str) -> str:
    """
    去掉失败任务前缀
    """
    if string.startswith(failed_sub_topic()):
        return string[len(failed_sub_topic()):]

    print("这个文件好像没有错误执行前缀")