import logging
from logging import handlers
import os
from utils.utils_time import get_cur_timestamp, get_cur_date
from utils.utils_file import get_base_path

LOF_FILE_DIR = "logs/"
LOG_FILE_SUFFIX = ".log"

log_name = get_cur_timestamp()

level = logging.INFO
dir_name = get_base_path() + LOF_FILE_DIR + get_cur_date() + "/"
os.makedirs(dir_name, exist_ok=True)
filename = dir_name + log_name + LOG_FILE_SUFFIX

filemode = 'w'
logger_format = '%(asctime)s %(filename)s:%(lineno)-4d %(levelname)-8s [%(funcName)s]%(message)s'

logging.basicConfig(
    level=level,
    filename=filename,
    filemode=filemode,
    format=logger_format,
)

logger = logging.getLogger(__name__)


def get_logger(conf: dict):

    global logger

    if logger is not None:
        return logger

    log_name = get_cur_timestamp()

    level = logging.INFO
    dir_name = get_base_path() + LOF_FILE_DIR + get_cur_date() + "/"
    os.makedirs(dir_name, exist_ok=True)
    filename = dir_name + log_name + LOG_FILE_SUFFIX

    filemode = 'w'
    logger_format = '%(asctime)s %(filename)s:%(lineno)-4d %(levelname)-8s [%(funcName)s]%(message)s'

    if "log_level" in conf:
        if conf['log_level'] == 'debug':
            level = logging.DEBUG

    logging.basicConfig(
        level=level,
        filename=filename,
        filemode=filemode,
        format=logger_format,
    )

    logger = logging.getLogger(__name__)

    return logger


def get_logger_v2():
    """
    暂不使用
    """
    # 创建一个日志器。提供了应用程序接口
    logger = logging.getLogger("自定义日志处理器")

    # 设置日志输出的最低等级,低于当前等级则会被忽略
    logger.setLevel(logging.INFO)

    # 创建日志输出路径
    log_path = get_logger_file_path()
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_name = get_cur_timestamp()
    log_path = log_path + log_name + LOG_FILE_SUFFIX

    # 创建格式器
    formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)-4d %(levelname)-8s [%(funcName)s]%(message)s')

    # 创建处理器：ch为控制台处理器，fh为文件处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 输出到文件
    fh = handlers.TimedRotatingFileHandler(
        filename=log_path,
        when='D',
        backupCount=100,
        encoding='utf-8')
    fh.setLevel(logging.INFO)

    # 设置日志输出格式
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 将处理器，添加至日志器中
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_logger_file_path() -> str:
    return get_base_path() + LOF_FILE_DIR


if __name__ == '__main__':

    logger = get_logger_v2()
    logger.debug("this to file")
    logger.info("this to stdout and file")