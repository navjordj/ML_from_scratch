import numpy as np 

class SquareLoss():

    def loss(self, X, y, y_pred):
        return (1/2)* np.power((y-y_pred), 2)

    def gradient(self, y, y_pred):
        """ 
        Derivative of loss function
        """
        return -(y-y_pred) 