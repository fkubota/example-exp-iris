from ipdb import set_trace as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import util


def run():
    print('start')

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
    run()


if __name__ == '__main__':
    exp()
