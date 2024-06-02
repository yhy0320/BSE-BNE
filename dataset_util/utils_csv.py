# import __init__
import pandas as pd

from dataset_util.common import *


def sample_csv_data(dataset_name: str, n: int) -> pd.DataFrame:
    """
    随机抽样指定csv文件中的n个样本到 dataframe 结构体
    """
    df = load_csv(construct_ds_path(dataset_name) + ds_file_path())
    sampled_df = df.sample(n)
    return sampled_df


def load_csv_data(dataset_name: str) -> pd.DataFrame:
    """
    装载csv文件到 dataframe 结构体
    """
    df = load_csv(construct_ds_path(dataset_name) + ds_file_path())
    return df


def load_csv_data_and_label(dataset_name: str) -> (pd.DataFrame, pd.DataFrame):
    """
    该函数根据数据集名称(dataset_name)返回数据集还有标签。
    """
    return split_data_and_label(load_csv_data(dataset_name))


def load_csv_data_and_label_with_sample(dataset_name: str, num: int) -> (pd.DataFrame, pd.DataFrame):
    """
    该函数根据数据集名称(dataset_name)返回数据集还有标签。
    """
    return sample_with_split_data_and_label(load_csv_data(dataset_name), num)


def load_csv(file_name: str) -> pd.DataFrame:
    """
    装载csv文件到 dataframe 结构体
    使用的是绝对路径
    """
    return pd.read_csv(file_name, index_col=None)


def save_csv(file_name: str, df: pd.DataFrame):
    """
    保存csv文件到指定目录
    """
    if not file_name.endswith('.csv'):
        file_name = file_name + CSV_SUFFIX
    df.to_csv(file_name, index=False)


def gen_csv_dataset_config(dataset_name: str):
    """
    生成数据集的配置文件

    :param dataset_name: 数据集名称
    """
    dataset_path = construct_ds_path(dataset_name)

    df = load_csv(dataset_path + DATASET_FILE_NAME + CSV_SUFFIX)
    # value_counts()用于计算DataFrame或Series中唯一值出现次数，返回一个包含唯一值及其出现次数的Series对象。
    count = df.iloc[:, -1].value_counts()
    print(len(count.keys()))
    config = {
        "instance": df.shape[0],
        "cluster": len(df.iloc[:, -1].value_counts().keys()),
        "dim": df.shape[1] - 1  # 减去标签
    }

    with open(dataset_path + CONFIG_FILE_NAME + JSON_SUFFIX, "w") as f:
        f.write(json.dumps(config))


def gen_csv_test_dataset_config(dataset_name: str):
    """
    生成数据集的配置文件

    :param dataset_name: 数据集名称
    """
    dataset_path = construct_test_path(dataset_name)
    df = load_csv(dataset_path + TEST_FILE_NAME + CSV_SUFFIX)
    count = df.iloc[:, -1].value_counts()
    print(len(count.keys()))
    config = {
        "instance": df.shape[0],
        "cluster": len(df.iloc[:, -1].value_counts().keys()),
        "dim": df.shape[1] - 1  # 减去标签
    }

    with open(dataset_path + CONFIG_FILE_NAME + JSON_SUFFIX, "w") as f:
        f.write(json.dumps(config))


def save_csv_dataset(df: pd.DataFrame, dataset_name: str, to_rsp_num=-1):
    dir_name = construct_ds_path(dataset_name)
    os.makedirs(dir_name, exist_ok=True)
    save_csv(dir_name + ds_file_path(), df)
    gen_csv_dataset_config(dataset_name)


def save_csv_test_dataset(df: pd.DataFrame, dataset_name: str):
    dir_name = construct_test_path(dataset_name)
    os.makedirs(dir_name, exist_ok=True)
    save_csv(dir_name + test_file_path(), df)
    gen_csv_test_dataset_config(dataset_name)


def decode_underline(x):
    if 'dataset_name' not in x:
        return x
    x['dataset_name'] = x['dataset_name'].replace('_', '\_')
    return x


def ratio_2_percent(x):
    if 'ratio' not in x:
        return x

    x['ratio'] = x['ratio'] * 100
    x['ratio'] = f'{x["ratio"]}\%'
    return x


def add_percent_symbol(x):
    if 'mean' not in x:
        return x

    x['mean'] = f'{x["mean"]}\%'
    return x


def csv_to_latex(df: pd.DataFrame):
    tabular = '|' + '|'.join(['c' for _ in range(df.shape[1])]) + '|'

    df = df.apply(decode_underline, axis=1)
    df = df.round(4)
    df = df.apply(ratio_2_percent, axis=1)
    df = df.apply(add_percent_symbol, axis=1)

    output = '\t' + ' & '.join([h.replace('_', '\_') for h in df.columns]) + ' \\\\ % Header \n'
    output = output + '\t\hline \n'
    for i in df.index:
        output = output + '\t' + ' & '.join([str(x) for x in df.loc[i].to_list()]) + f' \\\\ % row{i} \n'

    with open(get_base_path() + "dataset_util/tmpl/table.tex", 'r') as f:
        tmp = f.read()

    output = output + '\t\hline'
    s = tmp.format(tabular=tabular, content=output)
    print(s)
    return s


def generate_tests_file(dataset_name: str):
    """
    将已存在的文件夹分出测试数据集
    """

    dir_name = construct_test_path(dataset_name)
    # 是否已经创建文件夹
    if os.path.exists(dir_name):
        # 文件夹里面是否有文件
        if len(os.listdir(dir_name)) != 0:
            return

    df = load_csv_data(dataset_name)
    n = df.shape[0]
    sample = df.sample(int(n * 0.2))

    save_csv_test_dataset(sample, dataset_name)
    print(f'generate {dataset_name}`s test set')


def load_csv_test(dataset_name: str):
    """
    装载csv的测试文件到 dataframe 结构体
    """
    df = load_csv(construct_test_path(dataset_name) + test_file_path())
    return df
