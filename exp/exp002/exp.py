from ipdb import set_trace as st
import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import util


def run(run_name, config_update):
    print(f'\nstart: {run_name}')

    pwd = os.path.dirname(os.path.abspath(__file__))
    with open(f'{pwd}/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)

    # init_exp
    dir_save, config = util.init_exp(config, config_update, run_name)
    model_params = config['model']['params']
    split_params = config['split']

    # datast
    df, target = util.get_datasets()
    X_train, X_valid, y_train, y_valid = train_test_split(df, target, test_size=0.3, **split_params)

    # fit
    print(model_params)
    model = LogisticRegression(**model_params)
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
                max_iter: 10
        split:
            random_state: 1
        ''',
        '''
        model:
            params:
                max_iter: 30
        split:
            random_state: 2
        ''',
        '''
        model:
            params:
                max_iter: 60
        split:
            random_state: 3
        ''',
        '''
        model:
            params:
                max_iter: 80
        split:
            random_state: 4
        ''',
    ]

    for i_run, config_str in enumerate(list_config_str, 1):
        config_update = yaml.safe_load(config_str)
        run_name = f'run{str(i_run).zfill(3)}'
        run(run_name, config_update)


if __name__ == '__main__':
    exp()
