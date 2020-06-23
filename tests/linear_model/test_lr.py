import pytest
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.linear_model.logistic_regression import LogisticRegression

from ..datasets import binary_dataset


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