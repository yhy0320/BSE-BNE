import os
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from utils_csv import *
from utils import get_base_path


def load_arff_data(dir_name: str):

    f = open(dir_name)
    file_name = dir_name.split('/')[-1].split('.')[0]

    # arff.loadff用于加载ARFF文件。该方法返回两个值：data和meta，data是一个包含所有实例的列表，meta是一个包含属性信息的字典。
    arff_file = arff.loadarff(f)
    df = pd.DataFrame(arff_file[0])

    # LabelEncoder用于将分类变量转换为数字变量。
    class_le = LabelEncoder()
    df['CLASS'] = class_le.fit_transform(df['CLASS'].values)

    dir_name = get_base_path()
    dir_name = dir_name + DATASET_PATH + file_name + "/"

    os.makedirs(dir_name, exist_ok=True)

    save_csv(dir_name + "ds.csv", df)
