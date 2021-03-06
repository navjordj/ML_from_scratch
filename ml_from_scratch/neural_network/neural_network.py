import numpy as np 

from .layers import Layer

from tqdm import tqdm

# Neural Network API inspired by sequential model


def batch_iterator(X, y=None, batch_size=64):
    """
    Simple batch iterator 
    From https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/utils/data_manipulation.py
    """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]


class NeuralNetwork():


    def __init__(self, optimizer, loss_function):
        self.layers = []
        self.loss_function = loss_function()
        self.optimizer = optimizer
        self.errors = []


    def add_layer(self, layer: Layer):
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())

        # If the layer has weights that needs to be initialized 
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)

        # Add layer to the network
        self.layers.append(layer)

    def _forward(self, X, training=True):
        """ 
        Loops over every layer and does forward propagation
        Output of previous layer is input for the next
        """
        layer_output = X # First input is X

        for layer in self.layers:
            layer_output = layer.forward(layer_output, training)

        return layer_output

    def _train_batch(self, X, y):
        y_pred = self._forward(X)

        loss = np.mean(self.loss_function.loss(y, y_pred))

        loss_grad = self.loss_function.gradient(y, y_pred)

        self._backward(loss_grad=loss_grad)
        return loss

    def _backward(self, loss_grad):
        """
        Back propagate through the layers and updates the gradients according to the loss
        """

        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)


    def fit(self, X, y, n_epochs, batch_size):
        
        for _ in tqdm(range(n_epochs)):

            batch_errors = []
            for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size):
                loss = self._train_batch(X_batch, y_batch)
                batch_errors.append(loss)

            avg_batch_error = np.mean(batch_errors)

            self.errors.append(avg_batch_error)

        return self.errors

    def predict(self, X, argmax=True):
        """Forward pass through network for predicting
        """
        pred = self._forward(X, training=False)
        #print(pred)
        if argmax:
            return np.argmax(pred, axis=1)
        else:
            return pred