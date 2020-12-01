
import numpy as np 
import copy
import math

class Layer():

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, X, training):
        raise NotImplementedError()

    def set_input_shape(self, shape):
        self.input_shape = shape



class Dense(Layer):
    
    def __init__(self, units, input_shape=None):
        self.layer_input = None
        self.units = units 
        self.input_shape = input_shape
        self.W = None
        self.b = None

    def initialize(self, optimizer):
        """Initializes the weights and biases and the optimizer 
        """
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.units)) # TODO check limits
        self.b = np.zeros((1, self.units))

        self.W_optimizer = copy.copy(optimizer)
        self.b_optimizer = copy.copy(optimizer)

    def output_shape(self): # TODO make getter
        return (self.units, )


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

        # Update the layer weights
        self.W = self.W_optimizer.update(self.W, grad_w)
        self.b = self.b_optimizer.update(self.b, grad_b)

        accum_grad = accum_grad.dot(W.T)
        return accum_grad


class RNN(Layer):

    def __init__(self, units, vocab_size, input_shape):
        self.units = units 
        self.vocab_size = vocab_size
        self.input_shape = input_shape

        self.U = None
        self.V = None
        self.W = None

    def initialize(self, optimizer):
        pass

    def forward(self, X, training=True):
        pass 

    def backward(self, accum_grad):
        pass

class Activation(Layer):

    def __init__(self, activaton_function):
        self.activation_function = activaton_function()
        self.name = activaton_function.__name__

    
    def forward(self, X, training):
        self.layer_input = X
        return self.activation_function(X)

    def backward(self, accum_grad):
        return accum_grad * self.activation_function.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape
        

