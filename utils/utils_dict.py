import json
import __init__


def merge_dict(src: dict, dst: dict) -> dict:

    for k, v in src.items():
        dst[k] = v

    return dst


def list_to_map(kv_list: list) -> dict:
    """
    把 key-value数组转成map

    EXAMPLE
    --------
    l = [['key1', 'value1'], ['key2', 'value2']]
    m = list_to_map(l)
    print(m)
    {'key1': 'value1', 'key2': 'value2'}
    """
    if len(kv_list) == 0:
        return {}

    assert len(kv_list[0]) == 2, 'key-value数组必须是2维度的'

    dic = {}
    for kv in kv_list:
        dic[kv[0]] = kv[1]

    return dic


def order_dumps(dic: dict) -> str:
    jsn = json.dumps(sorted(dic.items(), key=lambda d: d[0]))
    return jsn


def order_loads(jsn: str) -> dict:
    jsn_list = json.loads(jsn)
    conf = list_to_map(jsn_list)
    return conf


def extract_special_config(src: dict, keys: list) -> dict:
    """
    提取特定 key 的配置文件
    """
    special_config = {}

    for key in keys:
        if key not in src:
            continue

        special_config[key] = src[key]

    return special_config


def format_dict(src: dict) -> str:
    ret = ""
    for k, v in src.items():
        ret += f'[{k} = {v}]'

    return ret


def save_conf(m: dict, abs_path: str):
    with open(abs_path, "w") as f:
        f.write(json.dumps(m))
