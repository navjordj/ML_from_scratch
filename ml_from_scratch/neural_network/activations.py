import numpy as np 

class Sigmoid():

    def __call__(self, X):
        return 1 / (1 + np.exp(X))


class ReLU():

    def __call__(self, X):
        return np.where(X >= 0, X, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

if __name__ == "__main__":
    x = Sigmoid()
    print(x(0))