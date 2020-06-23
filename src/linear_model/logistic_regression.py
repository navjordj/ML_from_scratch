import numpy as np

class LogisticRegression():

    def __init__(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self._w = np.random.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        # [bias, w1, w2, w2]

        self._cost = np.array([])

        for _ in range(self.epochs):
            net_input = self._net_input(X)
            output = self._activation(net_input)
            loss = (y - output)
            
            #Back propagate:
            self._w[1:] += self.learning_rate * X.T.dot(loss)
            self._w[0] += self.learning_rate * loss.sum()

            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self._cost = np.append(self._cost, cost)
            
        return self
    
    def predict(self, X):
        """Pass forward through the network
        """
        predictions = np.array([])
        probabilities = np.array([])

        if X.ndim == 1:
            p = self._activation(self._net_input(X))
            if  p >= 0.5:
                prediction = 1
            else:
                prediction = 0
            return prediction, p


        for x in X:
            p = self._activation(self._net_input(x))
            if  p >= 0.5:
                predictions = np.append(predictions, 1)
            else:
                predictions = np.append(predictions, 0)
            probabilities = np.append(probabilities, p)
        return predictions, probabilities

    def _net_input(self, X):
        return np.dot(X, self._w[1:]) + self._w[0]

    def _activation(self, z):
        return 1 / (1 + np.exp(-z))


if __name__ == "__main__":

    X = np.array([np.array([1, 2, 3]), np.array([4, 5, 6])])
    y = np.array([0, 1])

    lr = LogisticRegression(epochs=100, learning_rate=0.01)
    lr.fit(X, y)
    print(lr.predict(X))
    