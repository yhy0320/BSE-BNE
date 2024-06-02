import math
import matplotlib.pyplot as plt
import pandas as pd

from utils.utils_file import get_topic_result_dir_by_time
from .utils_csv import load_csv
from .utils_dataframe import averaging_result


def default_figure_config() -> dict:
    config = {
        'fig_size': (10, 10),
        'fig_name': 'fig.png',
    }
    return config


def extract_info(file_name: str):
    file_name = file_name.split("/")[1]
    file_name = file_name.split(".")[0]
    return file_name


def rate_col_ana(file_name):
    rate = []
    for i in range(10, 100, 10):
        rate.append([i / 100, (i + 10) / 100])

    df = pd.read_excel(file_name, index_col=0)
    plt.figure(figsize=(20, 100))
    pics_idx = 1

    for r in rate:
        rateDF = df.query(f'real_rate > {r[0]} & real_rate <= {r[1]}')

        change_rate = []
        count = 0
        satis = 0
        for i in range(rateDF.shape[0]):
            count += 1
            if rateDF.iloc[i, 2] >= rateDF.iloc[i, 3]:
                satis += 1
            change_rate.append(satis / count)

        t = extract_info(file_name)

        plt.subplot(20, 1, pics_idx)
        pics_idx += 1
        plt.scatter(rateDF.index, rateDF.iloc[:, 3], label="right", marker='x')
        plt.scatter(rateDF.index, rateDF.iloc[:, 2], label="left", marker='x')

        plt.title(f"rate: [{r[0]},{r[1]}), {satis} / {count} = {satis / count}")
        plt.legend()
        plt.tight_layout()
        # plt.savefig(f"./ana_{t}.png")
        # plt.show()

        plt.subplot(20, 1, pics_idx)
        pics_idx += 1
        plt.ylim(0.7, 1.05)
        plt.plot(rateDF.index, change_rate, label="rate")
        plt.title(f"rate: [{r[0]},{r[1]})")
        plt.legend()
        plt.tight_layout()

    plt.savefig(f"./ana_rate_{t}.png")
    plt.show()


def single_col_ana(file_name):
    df = pd.read_excel(file_name, index_col=0)
    df = df.query(f'bucket_num==10 & sample_rate > 0.10 & sample_rate <= 0.15')
    # print(aa)
    change_rate = []
    count = 0
    satis = 0
    for i in range(df.shape[0]):
        count += 1
        if df.iloc[i, 2] > df.iloc[i, 3]:
            satis += 1
        change_rate.append(satis / count)

    t = extract_info(file_name)

    plt.figure(figsize=(20, 5))
    plt.scatter(df.index, df.iloc[:, 3], label="right", marker='x')
    plt.scatter(df.index, df.iloc[:, 2], label="left", marker='x')

    plt.title(f"{satis} / {count} = {satis / count}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./ana_{t}.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, change_rate, label="rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./ana_rate_{t}.png")
    plt.show()


def draw_from_file(result_name: str):
    dir_name = get_topic_result_dir_by_time('run_DT', '2022-11-25', '13:21:45')
    df = load_csv(dir_name + 'result.csv')
    df = averaging_result(df, 'score', 5)
    print(df)


def draw_plot(df: pd.DataFrame, dir_name: str, x_col_name: str, y_col_name: str):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df[x_col_name], df[y_col_name])
    fig.savefig(dir_name + "draw_plot.png")


def draw_plot_by_ax(df: pd.DataFrame, ax, x_col_name: str, y_col_names: ...):
    for y_col_name in y_col_names:
        ax.plot(df[x_col_name], df[y_col_name])

    return ax


def draw_range_plot_by_ax(df: pd.DataFrame, ax, x, top, bottom):
    ax.fill_between(x, top, bottom, alpha=0.1, color='red')
    return ax


def square_figs(fig, fig_num: int):
    row = math.ceil(pow(fig_num, 0.5))  # 正方形

    print(f"这个图每行有{row}行")
    sub_plots = []
    for i in range(1, fig_num + 1):
        sub_plots.append(fig.add_subplot(row, row, i))

    return sub_plots
