from dataset_util.common import *
from dataset_util.utils_csv import *
from dataset_util.utils_dataframe import *


def test_gen_consts_file():
    gen_local_dataset_consts_file()


def test_wash_covid_dataset():
    """
    处理 covid 数据集
    """
    df = load_csv_data('covid')
    df = delete_column(df, ['DATE_DIED'])
    df = put_aim_to_tail(df, df.shape[1] - 2)
    print(df)

    save_csv_dataset(df, 'covid', to_rsp_num=100)


if __name__ == '__main__':
    test_gen_consts_file()
