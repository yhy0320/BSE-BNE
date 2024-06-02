import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

"""
https://blog.csdn.net/jiebaoshayebuhui/article/details/128576499

"""


def clear_NaN(df: pd.DataFrame) -> pd.DataFrame:
    imp_mean = SimpleImputer(
        missing_values=np.NaN,
        strategy='mean',
    )
    df = imp_mean.fit_transform(df)
    df = pd.DataFrame(df)
    return df


def label_encode(df: pd.DataFrame) -> pd.DataFrame:
    enc = preprocessing.LabelEncoder()
    df = enc.fit_transform(df)
    return df

