
import numpy as np 


class Layer():

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, X, training):
        raise NotImplementedError()



class Dense(Layer):
    
    def __init__(self, units, input_shape=None):
        self.units = units 
        self.input_shape = input_shape
        self.W = None
        self.b = None

    def initialize(self):
        self.W = np.random.uniform(-1, 1, self.units) # TODO check limits
        self.b = np.zeros((1, self.units))

    def forward(self):
        """Forward pass through the dense layer.
        Calculated using a dotproduct between input and weights and add bias
        """
        self.layer_input = X
        return np.dot(X, self.W) + self.b

    def backward(self):
        """Backward pass through the Dense layer
        """
        raise NotImplementedError()