import numpy as np 

def accuracy_score(pred, target):
    assert pred.shape == target.shape

    n = len(pred)

    correct = 0
    for i in range(n):
        if pred[i] == target[i]:
            correct += 1

    return correct/n