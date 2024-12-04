import numpy as np
from ..normalizer import Normalizer

class LogisticRegression:
    """A simple logistic regression model.
    """
    def __init__(self):
        """Initializes the model with the given parameters.
        """
        self.scaler_X = Normalizer()
        self.scaler_y = Normalizer()

    def sigmoid(self, z):
        """Returns the sigmoid of the given value.

        Args:
            z (float): the value to apply the sigmoid function to.

        Returns:
            float: the sigmoid of the given value.
        """
        return 1/(1 + np.exp(-z))
    
    def predict(self, X: np.ndarray) -> list:
        """Returns the model's prediction for the given input.

        Args:
            X (np.ndarray): an n-dimensional numpy array to be passed to the model,\
            representing a batch of inputs.

        Returns:
            list: the predicted output for each input.
        """
        X = self.scaler_X.transform(X)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        z = np.dot(X, self.w)
        probabilities = self.sigmoid(z)
        predictions = (probabilities >= 0.5).astype(int)
        return predictions.tolist()
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float):
        """Trains the model using the given training and validation data.

        Args:
            X (np.ndarray): the training production.
            y (np.ndarray): the validation production.
            epochs (int): the number of epochs to train the model.
            learning_rate (float): the learning rate for the gradient descent algorithm.
        """
        X = self.scaler_X.fit_transform(X)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        N, d = X.shape
        self.w = np.zeros(d)

        for _ in range(epochs):
            for i in range(N):
                z = np.dot(X[i], self.w)
                erro = y[i] - self.sigmoid(z)
                self.w = self.w + (learning_rate * erro * X[i])
        
        return
     