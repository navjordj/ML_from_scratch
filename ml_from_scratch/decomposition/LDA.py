
import numpy as np


# From DAT200 excercise

class LDA(object):

    def __init__(self, n_components=None, scaled_S_W=False):
        self.n_components = n_components
        self.scaled_S_W = scaled_S_W
        

    def fit(self, X, y):

        unique_labels = list(np.unique(y))
        

        mean_vecs = []
        for ind, label in enumerate(unique_labels):
            mean_vecs.append(np.mean(X[y == label], axis=0))        
        
        d = np.shape(X)[1] 
        if self.scaled_S_W == False:
        
            S_W = np.zeros((d, d))
            for label, mv in zip(unique_labels, mean_vecs):
                class_scatter = np.zeros((d, d)) 
                for row in X[y == label]:
                    row, mv = row.reshape(d, 1), mv.reshape(d, 1)
                    class_scatter += (row - mv).dot((row - mv).T)
                S_W += class_scatter                         
        
        else:            
            d = 13 
            S_W = np.zeros((d, d))
            for label, mv in zip(unique_labels, mean_vecs):
                class_scatter = np.cov(X[y == label].T)
                S_W += class_scatter
                    
        mean_overall = np.mean(X, axis=0)
        S_B = np.zeros((d, d))
        for i, mean_vec in enumerate(mean_vecs):
            n = X[y == i + 1, :].shape[0]
            mean_vec = mean_vec.reshape(d, 1)
            mean_overall = mean_overall.reshape(d, 1)
            S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
                
        self.eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
        
        eigen_pairs = [(np.abs(self.eigen_vals[i]), eigen_vecs[:, i])
                       for i in range(len(self.eigen_vals))]

        eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        
        if self.n_components is None:
            self.n_components = len(unique_labels) - 1
        
        w_list = []
        for comp in range(self.n_components):
            w_list.append(eigen_pairs[comp][1][:, np.newaxis].real)
        
        self.w = np.hstack(w_list)
            
        return self


    def transform(self, X):
        return X @ self.w

