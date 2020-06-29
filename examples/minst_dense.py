from ml_from_scratch.neural_network import NeuralNetwork
from ml_from_scratch.neural_network import GradientDescent
from ml_from_scratch.neural_network import SquareLoss
from ml_from_scratch.neural_network import Dense, Activation
from ml_from_scratch.neural_network import Softmax, ReLU
from ml_from_scratch.helpers import to_categorical, to_nominal
from ml_from_scratch.helpers import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn import datasets

import numpy as np



data = datasets.load_digits()
X = data["data"]
y = data["target"]

y = to_categorical(y.astype('int')) # One hot encoding

X = X.reshape(-1, 8*8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

optimizer = GradientDescent()

clf = NeuralNetwork(optimizer=optimizer, loss_function=SquareLoss)
clf.add_layer(Dense(input_shape=(8*8, 1), units=50))
clf.add_layer(Activation(ReLU))
clf.add_layer(Dense(units=20))
clf.add_layer(Activation(ReLU))
clf.add_layer(Dense(units=10))
clf.add_layer(Activation(Softmax))
clf.fit(X_train, y_train, n_epochs=200, batch_size=32)

pred = clf.predict(X_test)
print(accuracy_score(pred, to_nominal(y_test)))