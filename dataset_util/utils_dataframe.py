import pandas as pd
import numpy as np


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    # index属性是一个包含行索引标签的index对象
    # random.permutation()是一个用于生成随机排列的函数，可以返回随机排列的整数序列或数组，也可以对现有的序列或数组进行随机重排
    # reindex()方法用于重新索引行或列
    return df.reindex(np.random.permutation(df.index))


def averaging_result(df: pd.DataFrame, avg_col_name: str, avg_range, is_replace=False) -> pd.DataFrame:

    left_bound, right_bound = -1, 0
    if type(avg_range) == int:
        left_bound = -avg_range
        right_bound = 0
    elif type(avg_range) == list or type(avg_range) == np.ndarray:
        assert len(avg_range) >= 2, "这长度根本不够亲"
        left_bound = avg_range[0]
        right_bound = avg_range[1]
    else:
        assert len(avg_range) >= 2, "你这传的啥"

    count = df.shape[0]
    means = []
    # 忽略
    for i in df.index:
        means.append(
            df[avg_col_name][max(i + left_bound, 0):min(count - 1, i + right_bound) + 1].mean()
        )

    if is_replace:
        df[avg_col_name] = means
    else:
        df[f'mean_{avg_col_name}'] = means
    return df


def exchange_index_column(df: pd.DataFrame, idx: tuple):

    idx_value = df.columns.values       # 浅拷贝
    idx_value[idx[0]], idx_value[idx[1]] = idx_value[idx[1]], idx_value[idx[0]]


def reindex_column(df: pd.DataFrame, idx) -> pd.DataFrame:
    idx_value = df.columns.values  # 浅拷贝
    idx_value = idx_value[idx]
    df = df[idx_value]
    return df


def put_aim_to_tail(df: pd.DataFrame, idx: int):
    d = df.shape[1]
    assert idx < d, "这个索引有问题"

    index = [i for i in range(d)]
    index[idx], index[d - 1] = index[d - 1], index[idx]
    return reindex_column(df, index)


def put_label_to_tail(df: pd.DataFrame) -> pd.DataFrame:

    idx = [i for i in range(df.shape[1])]
    idx = idx[1:]
    idx.append(0)

    return reindex_column(df, idx)


def create_default_col(df: pd.DataFrame):
    """
    这个要有标签页
    """
    column_num = df.shape[1]
    column_names = [f'col{i}' for i in range(1, column_num)]
    column_names.append('label')
    df.columns = column_names


def delete_column(df: pd.DataFrame, cols: list) -> pd.DataFrame:

    df = df.drop(columns=cols)
    return df
