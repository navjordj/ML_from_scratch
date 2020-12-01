from ml_from_scratch.decomposition import PCA
from sklearn import decomposition

import numpy as np

def test_pca():
    X = np.array([
        [10, 10],
        [8, 10],
        [-5, 5.5],
        [-5.4, 5.5],
        [-20, -20],
        [-15, -20]
    ])

    pca_scratch = PCA(n_components=1)
    pca_scratch.fit(X)
    transform_scratch = pca_scratch.transform(X)

    pca_sk = decomposition.PCA(n_components=1)
    pca_sk.fit(X)
    transform_sk = pca_sk.transform(X)

    assert (transform_scratch == transform_sk).all()
