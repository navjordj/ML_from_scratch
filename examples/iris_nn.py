from ml_from_scratch.neural_network import NeuralNetwork
from ml_from_scratch.neural_network import Dense, Activation
from ml_from_scratch.neural_network import Softmax, ReLU
from ml_from_scratch.neural_network import GradientDescent
from ml_from_scratch.neural_network import SquareLoss
from ml_from_scratch.helpers import to_nominal, to_categorical, accuracy_score

from sklearn import datasets
from  sklearn.model_selection import train_test_split




data = datasets.load_iris()

X = data["data"]
y = data["target"]

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



optimizer = GradientDescent()

clf = NeuralNetwork(optimizer=optimizer, loss_function=SquareLoss)

clf.add_layer(Dense(input_shape=(4, 1), units=10))
clf.add_layer(Activation(ReLU))
clf.add_layer(Dense(units=5))
clf.add_layer(Activation(ReLU))
clf.add_layer(Dense(units=3))
clf.add_layer(Activation(Softmax))

clf.fit(X_train, y_train, n_epochs=100, batch_size=10)

pred = clf.predict(X_test)
print(accuracy_score(pred, to_nominal(y_test)))