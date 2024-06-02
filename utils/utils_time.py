from datetime import datetime


def get_cur_time():
    return datetime.now().strftime("%H:%M:%S")


def get_cur_date():
    return datetime.now().strftime("%Y-%m-%d")


def get_cur_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

