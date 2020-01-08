import os,json,re
from collections import namedtuple
from datetime import datetime, timedelta

HPARAMS_NAME = "params.json"

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def write_json(path, data):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)


def save_hparams(model_dir, hparams):
    param_path = os.path.join(model_dir, HPARAMS_NAME)

    info = eval(hparams.to_json(),{'false': False, 'true': True, 'null': None})
    write_json(param_path, info)

    print(" [*] MODEL dir: {}".format(model_dir))
    print(" [*] PARAM path: {}".format(param_path))

def load_json(path, as_class=False, encoding='euc-kr'):
    with open(path,encoding=encoding) as f:
        content = f.read()
        content = re.sub(",\s*}", "}", content)
        content = re.sub(",\s*]", "]", content)

        if as_class:
            data = json.loads(content, object_hook=\
                    lambda data: namedtuple('Data', data.keys())(*data.values()))
        else:
            data = json.loads(content)
    return data

def load_hparams(hparams, load_path, skip_list=[]):
    # log dir에 있는 hypermarameter 정보를 이용해서, hparams.py의 정보를 update한다.
    path = os.path.join(load_path, HPARAMS_NAME)

    new_hparams = load_json(path)
    hparams_keys = vars(hparams).keys()

    for key, value in new_hparams.items():
        if key in skip_list or key not in hparams_keys:
            print("Skip {} because it not exists".format(key))  #json에 있지만, hparams에 없다는 의미
            continue

        if key not in ['xxxxx',]:  # update 하지 말아야 할 것을 지정할 수 있다.
            original_value = getattr(hparams, key)
            if original_value != value:
                print("UPDATE {}: {} -> {}".format(key, getattr(hparams, key), value))
                setattr(hparams, key, value)