from sklearn.datasets import load_iris
import pandas as pd


def get_datasets():
    iris = load_iris()
    data = iris.data
    target = iris.target
    feature_names = iris.feature_names
    df = pd.DataFrame(data, columns=feature_names)
    return df, target
