import numpy as np
from typing import List

from models.model import Model


class LinearRegressionOLS(Model):
    """A simple linear regression model, using the Ordinary Least Squares (OLS) algorithm.
    """
    def __init__(self):
        self.w = None
        pass
    
    def predict(self, input_array: np.ndarray, **kwargs) -> List[str]:
        """Returns the model's prediction for the given input.

        Args:
            input_array (np.ndarray): an n-dimensional numpy array to be passed to the model,\
            representing a batch of inputs.

        Returns:
            List[float]: the predicted output for each input.
        """

        input_array = np.c_[np.ones(input_array.shape[0]), input_array]
        predictions = input_array @ self.w 
         
        return predictions.flatten().tolist()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Trains the model using the given training and validation data.

        Args:
            training_data (np.ndarray): the training production.
            validation_data (np.ndarray): the validation production.
            epochs (int): the number of epochs to train the model.
            learning_rate (float): the learning rate for the gradient descent algorithm.
        """
        X.reshape(-1,1)
        y.reshape(-1,1)

        X = np.c_[np.ones(X.shape[0]), X]

        self.w = np.linalg.pinv(X.T @ X) @ X.T @ y
        return
