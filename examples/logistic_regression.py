from ml_from_scratch.linear_model import LogisticRegression
import numpy as np

X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([0, 0, 1, 1])

X_train = np.array([[0, 0], [3.5, 3.5]])

clf = LogisticRegression(epochs=100, learning_rate=0.01)
clf.fit(X, y)

print(clf.predict(X_train))