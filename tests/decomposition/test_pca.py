
from sklearn import decomposition
from src.decomposition import PCA

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import pytest

from ..datasets import binary_dataset


def test_pca(binary_dataset):
    X_train, _ , _, _ = binary_dataset

    sklearn_PCA = decomposition.PCA()
    sklearn_PCA.fit(X_train)
    sklearn_transformed = sklearn_PCA.transform(X_train)

    scratch_PCA = decomposition.PCA()
    scratch_PCA.fit(X_train)
    scratch_transformed = scratch_PCA.transform(X_train)

    assert (sklearn_transformed == scratch_transformed).all()
