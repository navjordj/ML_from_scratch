from ml_from_scratch.neural_network import NeuralNetwork
from ml_from_scratch.neural_network import GradientDescent
from ml_from_scratch.neural_network import SquareLoss
from ml_from_scratch.neural_network import Dense, Activation
from ml_from_scratch.neural_network import Softmax, ReLU
from ml_from_scratch.helpers import to_categorical

from sklearn import datasets

import numpy as np




if __name__ == "__main__":

    #X = np.array([[1, 2], [10, 20], [2, 2], [11, 15]])
    #y = np.array([0, 1, 0, 1])

    data = datasets.load_digits()
    X = data["data"]
    y = data["target"]

    y = to_categorical(y.astype('int'))

    X = X.reshape(-1, 8*8)
    
    optimizer = GradientDescent()

    clf = NeuralNetwork(optimizer=optimizer, loss_function=SquareLoss)
    clf.add_layer(Dense(input_shape=(8*8, 1), units=50))
    clf.add_layer(Activation(ReLU))
    clf.add_layer(Dense(units=20))
    clf.add_layer(Activation(ReLU))
    clf.add_layer(Dense(units=10))
    clf.add_layer(Activation(Softmax))
    clf.fit(X, y, n_epochs=100, batch_size=32)
    
    print(np.argmax(clf.predict(X[15])), np.argmax(y[15]))