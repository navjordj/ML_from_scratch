import numpy as np


# From DAT200 excercise

class PCA(object):
    def __init__(self, n_components=None):
        self.n_components = n_components
        

    def fit(self, X):
        cov_mat = np.cov(X.T)
        self.eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        
        eigen_pairs = [(np.abs(self.eigen_vals[i]), eigen_vecs[:, i])
                       for i in range(len(self.eigen_vals))]

        eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        
        if self.n_components is None:
            self.n_components = np.shape(X)[1]
        
        w_list = []
        for comp in range(self.n_components):
            w_list.append(eigen_pairs[comp][1][:, np.newaxis])
        
        self.w = np.hstack(w_list)
        
        return self


    def transform(self, X):
        
        return X @ self.w
