from dataset_util.utils_mysql import *


def test_parse_sql():
    sql = ""
    with open('result.sql', 'r') as f:
        sql = f.read()
    parse_sql(sql)

    # print(sql)


if __name__ == '__main__':
    test_parse_sql()
