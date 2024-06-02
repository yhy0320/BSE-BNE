
# from dataset_util.utils_gen_data import make_circles
#
# make_circles(n_samples=100000)
import pandas as pd
from dataset_util.utils_dataframe import *
from dataset_util.utils_mysql import *
from dataset_util.common import *
from dataset_util.utils_draw import draw_plot, square_figs
from dataset_util.utils_mysql import load_from_mysql


def test1():

    """
    测试删除dataframe列
    """
    mydata = {
        '行程': [1234, 4235, 3214, 3421, 4312, 2341, 3214, 1432, 1423, 4211],
        '油耗': [14, 35, 32, 21, 41, 25, 32, 42, 12, 13],
        '时间': [0.3, 1.4, 1.8, 2.4, 3.1, 0.6, 0.7, 0.1, 0.5, 4.0],
        '优劣': [0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
    }
    df = pd.DataFrame(data=mydata)

    df = delete_column(df, ['油耗', '时间'])
    print(df)
    # draw_plot(df, '', '行程', '时间')


def test2():
    """
    测试更新数据库
    """
    ds_list = get_local_exist_dataset_list()
    conf_list = []
    for ds in ds_list:
        conf = load_dataset_conf(ds)
        conf['dataset_name'] = ds
        conf_list.append(conf)
    update_to_mysql(conf_list)


def test3():
    df = get_all_dataset_result_from_mysql()
    dataset_name = '10000_20_2_[-1000,1000]_dataset'
    # filter_dataset_name = df['dataset_name'] == '10000_20_2_[-1000,1000]_dataset'
    df = df.query(f'dataset_name == @dataset_name')
    print(df)


if __name__ == '__main__':
    test3()