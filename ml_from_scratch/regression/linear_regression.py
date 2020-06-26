import numpy as np 

class LinearRegression():
    """Gradient descent implementation of Linear Regression
    """

    def __init__(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate


    def fit(self, X, y):
        self._w = np.zeros(1+X.shape[1]) # +1 for bias
        self._cost = np.array([])

        for _ in range(self.epochs):
            output = self._net_input(X)
            loss = (y-output)

            self._w[1:] += self.learning_rate * X.T.dot(loss)
            self._w[0] += self.learning_rate * loss.sum()

            cost = (loss**2).sum() / 2.0 # RMSE
            self._cost = np.append(self._cost, cost)
        return self


    def _net_input(self, X):
        return np.dot(X, self._w[1:]) + self._w[0]

    def predict(self, X):
        return self._net_input(X)