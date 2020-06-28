import numpy as np 

from .layers import Layer

# Neural Network API inspired by sequential model


class NeuralNetwork():


    def __init__(self):
        self.layers = []


    def add_layer(self, layer: Layer):
        self.layers.append(layer)