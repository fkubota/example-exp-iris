from ipdb import set_trace as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def get_datasets():
    iris = load_iris()
    data = iris.data
    target = iris.target
    feature_names = iris.feature_names
    df = pd.DataFrame(data, columns=feature_names)
    return df, target


def run():
    print('start')
    df, target = get_datasets()
    X_train, X_valid, y_train, y_valid = train_test_split(df, target, test_size=0.33, random_state=42)


def exp():
    run()


if __name__ == '__main__':
    exp()
