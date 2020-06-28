
import numpy as np 
import copy


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

    def initialize(self, optimizer):
        """Initializes the weights and biases and the optimizer 
        """
        self.W = np.random.uniform(-1, 1, self.units) # TODO check limits
        self.b = np.zeros((1, self.units))

        self.W_optimizer = copy.copy(optimizer)
        self.b_optimizer = copy.copy(optimizer)

    def forward(self, X, training=True):
        """Forward pass through the dense layer.
        Calculated using a dotproduct between input and weights and add bias
        """
        self.layer_input = X
        return np.dot(X, self.W) + self.b

    def backward(self, accum_grad):
        """Backward pass through the Dense layer
        Weights are updated using the optimizer
        """

        W = self.W

        grad_w = self.layer_input.T.dot(accum_grad)
        grad_b = np.sum(accum_grad, axis=0, keepdims=True)

        self.W = self.W_optimizer.update(self.W, grad_w)
        self.b = self.b_optimizer.update(self.b, grad_b)

        accum_grad = accum_grad.dot(W.T)
        return accum_grad