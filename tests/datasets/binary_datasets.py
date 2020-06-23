import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

@pytest.fixture()
def binary_dataset():
    iris = load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)


    X_train = X_train[(y_train == 0) | (y_train == 1)]
    y_train = y_train[(y_train == 0) | (y_train == 1)]

    X_test = X_test[(y_test == 0) | (y_test == 1)]
    y_test = y_test[(y_test == 0) | (y_test == 1)]

    return X_train, X_test, y_train, y_test
