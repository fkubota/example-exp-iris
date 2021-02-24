import os
import yaml
import pandas as pd
from sklearn.datasets import load_iris


def get_datasets():
    iris = load_iris()
    data = iris.data
    target = iris.target
    feature_names = iris.feature_names
    df = pd.DataFrame(data, columns=feature_names)
    return df, target


def init_exp(config, config_update, run_name):
    # 保存ディレクトリの用意
    dir_save, exp_name = get_save_dir_exp(config, run_name)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    # configのupdateとconfig_updateの保存
    deepupdate(config, config_update)
    with open(f'{dir_save}/config_update.yml', 'w') as path:
        yaml.dump(config_update, path)
    return dir_save, config


def get_save_dir_exp(config, run_name):
    _dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = _dir.split('/')[-1]
    dir_save_exp = f'{config["path"]["dir_save"]}{exp_name}/{run_name}'
    return dir_save_exp, exp_name


def deepupdate(dict_base, other):
    '''
    ディクショナリを再帰的に更新する
    ref: https://www.greptips.com/posts/1242/
    '''
    for k, v in other.items():
        if isinstance(v, dict) and k in dict_base:
            deepupdate(dict_base[k], v)
        else:
            dict_base[k] = v

