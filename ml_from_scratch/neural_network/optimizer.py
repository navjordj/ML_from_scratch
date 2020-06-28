import numpy as np

class GradientDescent():
    """Stochastic Gradient Descent implementation
    """

    def __init__(self, learning_rate=0.01):
        """
        Args:
            learning_rate (float, optional): Learning rate for the opimization algorithm. Defaults to 0.01.
        """
        self.learning_rate = learning_rate
        self.w_update = None


    def update(self, w, grad_wrt): # TODO add momentum for more efficent optimization
        if self.w_update is None:
            self.w_update = np.zeros(np.shape(w))

        return w - self.learning_rate * self.w_update

