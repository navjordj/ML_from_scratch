
import numpy as np 
import copy
import math

from ..helpers import image_to_column, determine_padding

class Layer():

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, X, training):
        raise NotImplementedError()

    def backward(self, X, accum_grad):
        raise NotImplementedError()

    def initialize(self, optimizer):
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


class Conv2D(Layer):

    def __init__(self, n_filters, filter_shape, stride=1, padding=1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding

    
    def initialize(self, optimizer):
        f_height, f_width = self.filter_shape.shape # 2D filter
        n_channels = self.input_shape[0]

        limit = 1 / math.sqrt(np.prod(self.filter_shape))

        self.W = np.random.uniform(-limit, limit, size=(self.n_filters, n_channels, f_height, f_width))
        self.b = np.zeros((self.n_filters, 1))


        self.W_optimizer = copy.copy(optimizer)
        self.b_optimizer = copy.copy(optimizer)


    def forward(self, X, training=True):
        batch_size, channels, height, width = X.shape
        self.layer_input = X

        self.X_col = image_to_column(X, self.filter_shape, self.stride, output_shape=self.padding)
        self.W_Col = self.W.reshape((self.n_filters, -1))

        output = self.W_Col.dot(self.X_col)  + self.b

        output = output.reshape(self.output_shape() + (batch_size, 1))
        return output.transpose(3, 0, 1, 2)


    def output_shape(self):
        n_channels, height, width = self.input_shape

        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width) 

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
        

