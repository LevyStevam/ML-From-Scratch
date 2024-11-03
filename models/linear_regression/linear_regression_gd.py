import numpy as np
import matplotlib.pyplot as plt
from typing import List

from models.model import Model


class LinearRegressionGD(Model):
    """A simple linear regression model, using gradient descent to train the model.
    """
    def __init__(self, w0: float = 0.0, w1: float = 0.0):
        """Initializes the model with the given parameters.

        Args:
            w0 (float): the intercept of the model.
            w1 (float): the slope of the model.
        """
        self.w0 = w0
        self.w1 = w1

    def predict(self, input_array: np.ndarray, **kwargs) -> List[float]:
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

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float, plot_learning_curve: bool = False):
        """Trains the model using the given training and validation data.

        Args:
            X_train (np.ndarray): the training production.
            y_train (np.ndarray): the validation production.
            epochs (int): the number of epochs to train the model.
            learning_rate (float): the learning rate for the gradient descent algorithm.
            plot_learning_curve (bool): if True, plots the learning curve at the end of training.
        """
        mse_history = []  

        for _ in range(epochs):
            error_w0 = 0.0
            error_w1 = 0.0

            for feature, y_true in zip(X_train, y_train):
                y_pred = self.w0 + self.w1 * feature
                error_w0 += (y_true - y_pred)
                error_w1 += (y_true - y_pred) * feature

            self.w0 += learning_rate * error_w0 / len(X_train)
            self.w1 += learning_rate * error_w1 / len(X_train)

            y_pred = self.predict(X_train)
            mse_history.append(self.mse(y_train, np.array(y_pred)))

        if plot_learning_curve:
            self.__plot_learning_curve(mse_history)

    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the mean squared error between the predicted and actual values.

        Args:
            y_true (np.ndarray): the actual values.
            y_pred (np.ndarray): the predicted values.

        Returns:
            float: the mean squared error between the predicted and actual values.
        """
        mse = np.mean((y_true - y_pred)**2)
        return mse

    def __plot_learning_curve(self, mse_history: List[float]):
        
        """Plota a curva de aprendizado invertida para mostrar a parte estável na parte inferior.
        
        Args:
            mse_history (List[float]): List of MSE values recorded during training.
        """
        plt.plot(mse_history)
        plt.xlabel("Épocas")
        plt.ylabel("MSE")
        plt.title("Curva de Aprendizado")
        plt.gca().invert_yaxis()  
        plt.show()
