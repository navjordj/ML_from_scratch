import numpy as np
import pytest

from sklearn import linear_model
from src.regression.linear_regression import LinearRegression
from sklearn.datasets import make_regression

@pytest.fixture
def regression_dataset():
    X, y = make_regression(n_samples=100, n_features=2)
    return X, y

def test_linear_regression(regression_dataset):
    
    X, y = regression_dataset

    sklearn_lr = linear_model.LinearRegression()
    sklearn_lr.fit(X, y)
    sklearn_preds = sklearn_lr.predict(X)

    scratch_lr = LinearRegression(epochs=100, learning_rate=0.01)
    scratch_lr.fit(X, y)
    scratch_preds = scratch_lr.predict(X)

    assert np.allclose(sklearn_preds, scratch_preds)