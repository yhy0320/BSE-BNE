import pandas as pd

from dataset_util.utils_dataframe import *
from dataset_util.utils_csv import *
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../letter-recognition.data', index_col=None, header=None)

df = put_label_to_tail(df)

create_default_col(df)

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

save_csv_dataset(df, 'letter_reco.csv')
