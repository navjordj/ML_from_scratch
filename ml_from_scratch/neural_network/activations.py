import numpy as np 

class Sigmoid():

    def __call__(self, X):
        return 1 / (1 + np.exp(X))


class ReLU():

    def __call__(self, X):
        return np.where(X >= 0, x, 0)

if __name__ == "__main__":
    x = Sigmoid()
    print(x(0))