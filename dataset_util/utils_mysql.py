import pandas as pd
import pymysql
from sqlalchemy import create_engine


# model 生成代码
# sqlacodegen mysql+pymysql://root:123456@192.168.31.23:3306/bse_bne
# --tables all_data_result --outfile all_data_result.py


def get_db_info(database='tune') -> dict:

    db_info = {
        'host': '192.168.31.23',
        'user': 'root',
        'port': 3306,
        'password': '123456',
        'database': database,
        'charset': 'utf8',
    }

    return db_info


def get_mysql_engine(database='tune'):
    db_info = get_db_info(database)

    engine = create_engine(
        'mysql+pymysql://%(user)s:%(password)s@%(host)s:%(port)d/%(database)s?charset=utf8' % db_info,
        # encoding='utf-8'
    )

    return engine


def save_to_mysql(df: pd.DataFrame, table_name: str, database='tune'):

    engine = get_mysql_engine(database)

    df.to_sql(table_name, engine, if_exists='append', index=False)


def load_from_mysql():

    engine = get_mysql_engine()

    sql = "select * from global_result"

    df = pd.read_sql(sql, engine)
    return df


def update_to_mysql(conf_list: list):

    sql = "update global_result set dimension=%(dim)d where dataset_name='%(dataset_name)s'"

    db_info = get_db_info()

    conn = pymysql.connect(**db_info)
    cursor = conn.cursor()

    for conf in conf_list:
        ss = sql % conf
        cursor.execute(sql % conf)

    conn.commit()
    cursor.close()
    conn.close()


def get_all_dataset_result_from_mysql():
    engine = get_mysql_engine()

    sql = "select * from all_dataset_result order by create_time desc"

    df = pd.read_sql(sql, engine)
    return df


def query(conf: dict):
    engine = get_mysql_engine(conf['database'])

    sql = "select * from all_data_result"
    if len(conf) > 0:
        sql += " where "
        conditions = []
        for k, v in conf.items():
            if k == 'database':
                continue
            conditions.append(f"{k} = '{v}'")

        sql += ' AND '.join(conditions)

    sql += " order by create_time desc"

    df = pd.read_sql(sql, engine)
    return df


def get_target_dataset_result_from_mysql(dataset_name: str) -> pd.DataFrame:
    df = get_all_dataset_result_from_mysql()

    if df.shape[0] == 0:
        return pd.DataFrame()

    if df.shape[0] == 1:
        df['cost_time'] = float(df['cost_time'])
        if df['cost_time'] != dataset_name:
            return pd.DataFrame()

    elif df.shape[0] > 1:
        df['cost_time'] = df['cost_time'].apply(pd.to_numeric)
        df = df.query(f'dataset_name == @dataset_name')

    return df


def get_target_dataset_result_from_mysql_v2(dataset_name: str, database: str) -> pd.DataFrame:
    df = query({
        'database': database,
        'dataset_name': dataset_name,
    })

    if df.shape[0] == 0:
        return pd.DataFrame()

    if df.shape[0] == 1:
        df['cost_time'] = float(df['cost_time'])

    elif df.shape[0] > 1:
        df['cost_time'] = df['cost_time'].apply(pd.to_numeric)

    return df
