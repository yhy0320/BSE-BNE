from dataset_util.utils_mysql import *
from utils.utils_dict import order_dumps, extract_special_config
from dataset_util.utils_csv import csv_to_latex
from dataset_util.utils_dataframe import delete_column

# 插入数据所需的keys
thesis_special_keys = [
    'params',
    'dataset_name',
    'sample',
    'instance',
    'params',
    'dimension',
    'cost_time',
    'p_cost_time',
    'score',
    'p_score',
    'mean',
    'var',
    'file_path',
    'run_id',
    'tag',
    'ratio',
]


def construct_thesis_dataframe():
    """
    result 结构体
    """
    df = pd.DataFrame(columns=thesis_special_keys)

    return df


def construct_dataframe_row(conf: dict):
    """
    构造 mysql 插入数据的结构
    """
    df = construct_thesis_dataframe()
    row = extract_special_config(conf, thesis_special_keys)
    df = df.append(row, ignore_index=True)
    return df


def insert_thesis(conf: dict):
    """
    插入一条新的数据
    """
    engine = get_mysql_engine('bse_bne')
    df = construct_dataframe_row(conf)
    df.to_sql('thesis', engine, if_exists='append', index=False)


def query(conf: dict):
    """
    查询
    """
    engine = get_mysql_engine('bse_bne')

    sql = "select * from thesis"

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


def toLatex(conf: dict):
    df = query(conf)
    df = delete_column(df, ['instance', 'create_time', 'id'])
    csv_to_latex(df)