from dataset_util.utils_mysql import *
from utils.utils_dict import order_dumps, extract_special_config


# 插入数据所需的keys
result_special_keys = [
    'dataset_name',
    'instance',
    'dimension',
    'cost_time',
    'params',
    'metrics',
    'file_path',
    'run_id',
    'config',
    'batch_run_id',
]


def construct_result_dataframe():
    """
    result 结构体
    """
    df = pd.DataFrame(columns=[
        'dataset_name',
        'instance',
        'dimension',
        'cost_time',
        'params',
        'metrics',
        'file_path',
        'run_id',
        'config',
        'batch_run_id',
    ])

    return df


def construct_dataframe_row(conf: dict):

    """
    构造 mysql 插入数据的结构
    """

    df = construct_result_dataframe()

    row = extract_special_config(conf, result_special_keys)

    df = df.append(row, ignore_index=True)

    return df


def insert_bne_result(conf: dict):
    """
    插入一条新的数据
    """
    engine = get_mysql_engine('bse_bne')
    df = construct_dataframe_row(conf)
    df.to_sql('result', engine, if_exists='append', index=False)


def query(conf: dict):
    """
    查询
    """
    engine = get_mysql_engine('bse_bne')

    sql = "select * from result"

    if len(conf) > 0:
        sql += " where "
        conditions = []
        for k, v in conf.items():
            conditions.append(f"{k} = '{v}'")

        sql += ' AND '.join(conditions)

    sql += " order by create_time desc"

    df = pd.read_sql(sql, engine)
    # print(df)
    return df


def favorite(run_id: str):
    assert run_id != "", "这啥 run_id"

    sql = f"update result set tag = 'star' where run_id = '{run_id}'"

    db_info = get_db_info('bse_bne')

    conn = pymysql.connect(**db_info)
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()


def cancel_favorite(run_id: str):
    assert run_id != "", "这啥 run_id"

    sql = f"update result set tag = '' where run_id = '{run_id}'"

    db_info = get_db_info('bse_bne')

    conn = pymysql.connect(**db_info)
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()


def update(db_where: str, db_set: str):

    sql = f"update result set {db_set} where {db_where};"

    print(sql)

    db_info = get_db_info('bse_bne')

    conn = pymysql.connect(**db_info)
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
   pass
