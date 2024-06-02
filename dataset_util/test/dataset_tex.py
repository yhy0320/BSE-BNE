import json
import pandas as pd

from dataset_util.utils_mysql import *
from dataset_util.utils_csv import csv_to_latex, load_csv
from dataset_util.common import *
from dataset_util.utils_dataframe import *


def test1():
    """
    加载本地 heihei.csv 数据转成tex
    """
    df = load_csv('./heihei.csv')
    df = df.drop(columns=['instance'])
    df['mean'] = df['mean'].abs() * 100
    df['var'] = df['var'] * 100
    csv_to_latex(df)


def test2():
    """
    将 heihei.csv 数据转成 flask 专用返回格式
    """
    df = load_csv('./heihei.csv')
    m = df.to_dict(orient='records')
    print(json.dumps(m))
    print(m)


def test3():

    df = load_from_mysql()
    df = delete_column(df, ['instance', 'create_time', 'id'])
    # print(df)
    df[['all_cost_time', 'rsp_cost_time', 'sample_ratio', 'diff_mean', 'diff_var', 'cost_time_ratio']] = df[['all_cost_time', 'rsp_cost_time', 'sample_ratio', 'diff_mean', 'diff_var', 'cost_time_ratio']].apply(pd.to_numeric)
    df = df.round(2)
    df = df.rename(columns={
        'all_cost_time': 'T_a',
        'rsp_cost_time': 'T_r',
        'dimension': 'd',
        'dataset_name': 'name',
        'sample_size': 'N_s',
        'sample_ratio': 'R_s',
        'diff_mean': 'mean',
        'diff_var': 'var',
        'cost_time_ratio': 'R_c',
    })
    csv_to_latex(df)

if __name__ == '__main__':

    test3()