from dataset_util.dataset_name_consts import *
from dataset_util.utils_csv import load_csv_data, save_csv_dataset
from dataset_util.utils_preprocessing import *
from dataset_util.utils_dataframe import create_default_col


def test_deal_with_Rice_MSC():
    df = load_csv_data(Dataset_name_Rice_MSC)
    # result = df.isna().values.any()
    # print(result)

    df = clear_NaN(df)
    # result = df.isna().values.any()
    # print(result)

    create_default_col(df)
    print(df)

    save_csv_dataset(df, Dataset_name_Rice_MSC, to_rsp_num=20)


if __name__ == '__main__':
    test_deal_with_Rice_MSC()