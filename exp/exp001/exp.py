from ipdb import set_trace as st
import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import util


def run(run_name, config_update):
    print('start')

    pwd = os.path.dirname(os.path.abspath(__file__))
    with open(f'{pwd}/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)

    # init_exp
    dir_save, config = util.init_exp(config, config_update, run_name)

    # datast
    df, target = util.get_datasets()
    X_train, X_valid, y_train, y_valid = train_test_split(df, target, test_size=0.5, random_state=42)

    # fit
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # pred
    X_valid_pred = model.predict(X_valid)

    # eval
    acc_valid = accuracy_score(y_valid, X_valid_pred)
    print(f'acc_valid: {acc_valid:.5f}')



def exp():

    list_config_str = [
    '''
    model:
        params:
            max_depth: 3
    ''',
    '''
    model:
        params:
            max_depth: 10
    ''',
    '''
    model:
        params:
            max_depth: 20
    ''',
    ]


    for i_run, config_str in enumerate(list_config_str, 1):
        config_update = yaml.safe_load(config_str)
        run_name = f'run{str(i_run).zfill(3)}'
        run(run_name, config_update)

if __name__ == '__main__':
    exp()
