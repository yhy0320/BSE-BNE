import pandas as pd

from dataset_util.utils_rsp import select_rsp_list, to_rsp
from dataset_util.dataset_name_consts import *


def rsp_test():
    rsp, _ = select_rsp_list('10000_20_2_[-1000,1000]_dataset', 1)
    assert type(rsp[0]) == pd.DataFrame, "错啦错啦"

    print(rsp)

    rsp, _ = select_rsp_list('10000_20_2_[-1000,1000]_dataset', 2)
    assert len(rsp) > 1, "错啦错啦"

    print(rsp)


if __name__ == '__main__':
    # rsp_test()
    to_rsp(Dataset_name_SUSY, 100)