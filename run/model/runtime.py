from dataset_util.utils_mysql import *
from utils.utils_dict import order_dumps, extract_special_config

TABLE_NAME = "runtime"

# 插入数据所需的keys
runtime_special_keys = [
    'run_id',
    'batch_run_id',
    'config',
    'seed',
]


def construct_dataframe():
    """
    runtime 结构体
    """
    df = pd.DataFrame(columns=[
        'run_id',
        'batch_run_id',
        'config',
        'seed',
    ])

    return df


def dict_to_dataframe(conf: dict):
    """
    构造 mysql 插入数据的结构
    """
    df = construct_dataframe()
    row = extract_special_config(conf, runtime_special_keys)
    df = df._append(row, ignore_index=True)

    return df


def insert(conf: dict):
    """
    插入一条新的数据
    """
    engine = get_mysql_engine('bse_bne')
    df = dict_to_dataframe(conf)
    df.to_sql(TABLE_NAME, engine, if_exists='append', index=False)


def query(conf: dict):
    """
    查询
    """
    engine = get_mysql_engine('bse_bne')

    sql = f"select * from {TABLE_NAME}"

    if len(conf) > 0:
        sql += " where "
        conditions = []
        for k, v in conf.items():
            conditions.append(f"{k} = '{v}'")

        sql += ' AND '.join(conditions)

    sql += " order by create_time desc"

    df = pd.read_sql(sql, engine)

    return df


def __execute(sql: str):
    db_info = get_db_info('bse_bne')

    conn = pymysql.connect(**db_info)
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()


def favorite(run_id: str):
    assert run_id != "", "这啥 run_id"
    sql = f"update {TABLE_NAME} set tag = 'star' where run_id = '{run_id}'"
    __execute(sql)


def cancel_favorite(run_id: str):
    assert run_id != "", "这啥 run_id"
    sql = f"update result set tag = '' where run_id = '{run_id}'"
    __execute(sql)


def update(db_where: str, db_set: str):
    sql = f"update {TABLE_NAME} set {db_set} where {db_where};"
    __execute(sql)


def mark_success(run_id: str):
    sql = f"update {TABLE_NAME} set code=1 where run_id = '{run_id}';"
    __execute(sql)
