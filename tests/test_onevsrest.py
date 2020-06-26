import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from sklearn import multiclass
from ml_from_scratch.onevsrest import OneVSRest
from ml_from_scratch.linear_model import LogisticRegression

import pytest

@pytest.mark.skip("Incorrect")
def test_onevsrest():
    X = np.array([
        [10, 10],
        [8, 10],
        [-5, 5.5],
        [-5.4, 5.5],
        [-20, -20],
        [-15, -20]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])


    sklearn_clf = multiclass.OneVsRestClassifier(linear_model.LogisticRegression()).fit(X, y)
    sklearn_preds = sklearn_clf.predict(np.array([[-19, -20], [9, 9], [-5, 5]]))

    scratch_clf = OneVSRest(LogisticRegression, epochs=100, learning_rate=0.01)
    scratch_clf.fit(X, y)
    scrath_preds = scratch_clf.predict(np.array([[-19, -20], [9, 9], [-5, 5]]))

    assert (sklearn_preds == scrath_preds).all() 
