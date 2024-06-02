import pandas as pd

from dataset_util.utils_rsp import *
from dataset_util.utils_dataframe import *
from dataset_util.utils_csv import *


def test_df() -> pd.DataFrame:
    df = pd.DataFrame(data={
        '行程': [1234, 4235, 3214, 3421, 4312, 2341, 3214, 1432, 1423, 4211],
        '油耗': [14, 35, 32, 21, 41, 25, 32, 42, 12, 13],
        '时间': [0.3, 1.4, 1.8, 2.4, 3.1, 0.6, 0.7, 0.1, 0.5, 4.0],
        '优劣': [0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
    })
    return df


def test_create_default_col():
    df = pd.read_csv("../train/X_train.txt", index_col=None, header=None, delim_whitespace=True)
    create_default_col(df)
    label = pd.read_csv("../train/y_train.txt", index_col=None, header=None)
    df['label'] = label[0]

    save_csv_dataset(df, 'UCI_HAR')


def test_gen_tests_file():
    # ds_list = get_local_exist_dataset_list()
    #
    # for ds in ds_list:
    #     gen_tests_file(ds)
    generate_tests_file('500000_10_10_[-1000,1000]_dataset')
    generate_tests_file('1000000_200_2_[-50000,50000]_dataset')
    generate_tests_file('circle')
    generate_tests_file('Covtype')
    generate_tests_file('Dry_Bean')
    generate_tests_file('letter_reco')
    generate_tests_file('PA')
    generate_tests_file('microbes')
    generate_tests_file('mnist')


def test_gen_UCI_HAR_test_file():
    generate_tests_file(Dataset_name_UCI_HAR)


if __name__ == '__main__':
    test_gen_UCI_HAR_test_file()