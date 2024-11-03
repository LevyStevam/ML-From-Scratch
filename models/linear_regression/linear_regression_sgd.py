import numpy as np
from typing import List

from models.model import Model


class LinearRegressionSGD(Model):
    """A simple linear regression model, using stochastic gradient descent to train the model.
    """
    def __init__(self, w0: float = 0.0, w1: float = 0.0):
        """Initializes the model with the given parameters.

        Args:
            w0 (float): the intercept of the model.
            w1 (float): the slope of the model.
        """
        self.w0 = w0
        self.w1 = w1
        pass
    
    def predict(self, input_array: np.ndarray, **kwargs) -> List[str]:
        """Returns the model's prediction for the given input.

        Args:
            input_array (np.ndarray): an n-dimensional numpy array to be passed to the model,\
            representing a batch of inputs.

        Returns:
            List[float]: the predicted output for each input.
        """
        predictions = []
        
        for feature in input_array:
            y_pred = self.w0 + self.w1 * feature
            predictions.append(y_pred)
        
        return predictions

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float):
        """Trains the model using the given training and validation data.

        Args:
            training_data (np.ndarray): the training production.
            validation_data (np.ndarray): the validation production.
            epochs (int): the number of epochs to train the model.
            learning_rate (float): the learning rate for the gradient descent algorithm.
        """
        for _ in range(epochs):
            
            for feature, y_true in zip(X, y):
                y_pred = self.w0 + self.w1 * feature
                error = y_true - y_pred
                
                self.w0 += learning_rate * error
                self.w1 += learning_rate * error * feature
        return

    def mse(self, y_train: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the mean squared error between the predicted and actual values.

        Args:
            y_train (np.ndarray): the actual values.
            y_pred (np.ndarray): the predicted values.

        Returns:
            float: the mean squared error between the predicted and actual values.
        """
        mse = np.mean((y_train - y_pred)**2)
        return mse