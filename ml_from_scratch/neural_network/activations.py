import numpy as np 

class Sigmoid():

    def __call__(self, X):
        return 1 / (1 + np.exp(X))


if __name__ == "__main__":
    x = Sigmoid()
    print(x(0))