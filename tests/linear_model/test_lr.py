import pytest
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.linear_model.logistic_regression import LogisticRegression



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



def test_lr(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset

    # Sklearn:
    sklearn_lr = linear_model.LogisticRegression(penalty="none")
    sklearn_lr.fit(X_train, y_train)
    sklearn_preds = sklearn_lr.predict(X_test)
    
    scratch_lr = LogisticRegression(epochs=100, learning_rate=0.01)
    scratch_lr.fit(X_train, y_train)
    scratch_preds = scratch_lr.predict(X_test)

    assert (sklearn_preds == scratch_preds).all()