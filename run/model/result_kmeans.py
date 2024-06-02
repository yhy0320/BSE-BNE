from dataset_util.utils_mysql import *
from utils.utils_dict import order_dumps, extract_special_config

TABLE_NAME = "result_kmeans"

# 插入数据所需的keys
result_kmeans_special_keys = [
    'dataset_name',
    'instance',
    'run_id',
    'batch_run_id',
    'k',
    'seed',
    'homogeneity_score',
    'completeness_score',
    'v_measure_score',
    'adjusted_rand_score',
    'adjusted_mutual_info_score',
    'fowlkes_mallows_score',
    'rand_score',
    'mutual_info_score',
    'normalized_mutual_info_score',
]


def dict_2_dataframe(conf: dict):
    """
    构造 mysql 插入数据的结构
    """
    df = pd.DataFrame(columns=result_kmeans_special_keys)
    row = extract_special_config(conf, result_kmeans_special_keys)
    df = df.append(row, ignore_index=True)

    return df


def insert(conf: dict):
    """
    插入一条新的数据
    """
    engine = get_mysql_engine('bse_bne')
    df = dict_2_dataframe(conf)
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


def update(db_where: str, db_set: str):
    sql = f"update {TABLE_NAME} set {db_set} where {db_where};"
    __execute(sql)

