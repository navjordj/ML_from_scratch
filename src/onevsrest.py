# Implement One VS all
import numpy as np 
import copy

class OneVSRest():

    def __init__(self, model, epochs, learning_rate):
        self.model = model(epochs, learning_rate)
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.classifiers = {}
        self.unique_classes = None

    def fit(self, X, y):
        self.unique_classes = np.unique(y)
        n_classes = self.unique_classes.shape[0]


        for target in self.unique_classes:
            idx_target = (y == target)
            
            y_target = y.copy()
            y_target[idx_target] = 1
            y_target[np.invert(idx_target)] = 0

            classifier = copy.copy(self.model)
            classifier.fit(X, y_target)
            self.classifiers[target] = classifier


    def predict(self, X):
        predictions = []
        for x in X:
            print(x)
            max_prob = -1
            pred = None
            for target in self.unique_classes:
                pred, prob = self.classifiers[target].predict(x)
                if prob > max_prob:
                    max_prob = pred
                    pred = target
            predictions.append(pred)

        return predictions
