import os
import matplotlib.pyplot as plt
import pandas as pd
from utils.utils_file import get_base_path
from sklearn.datasets import make_circles as circles
from .utils_csv import save_csv_dataset


def make_circles(n_samples=500, factor=0.5, noise=0.05):
    c = circles(n_samples, factor=factor, noise=noise)
    df = pd.DataFrame()
    df['col1'] = c[0][:, 0]
    df['col2'] = c[0][:, 1]
    df['label'] = c[1]
    save_csv_dataset(df, f'circle')

