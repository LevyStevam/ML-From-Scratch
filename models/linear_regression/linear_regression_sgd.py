import numpy as np
import matplotlib.pyplot as plt

from typing import List
from sklearn.utils import shuffle

from models.model import Model


class LinearRegressionSGD(Model):
    """A simple linear regression model, using stochastic gradient descent to train the model.
    """
    def __init__(self):
        """Initializes the model with the given parameters.

        Args:
            w0 (float): the intercept of the model.
            w1 (float): the slope of the model.
        """
        self.w0 = 0
        self.w1 = 0
        self.mse_history = [] 
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

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float, plot_learning_curve: bool = False):
        """Trains the model using the given training and validation data.

        Args:
            training_data (np.ndarray): the training production.
            validation_data (np.ndarray): the validation production.
            epochs (int): the number of epochs to train the model.
            learning_rate (float): the learning rate for the gradient descent algorithm.
        """

        for _ in range(epochs):
            X, y = shuffle(X, y)
            
            for feature, y_true in zip(X, y):
                y_pred = self.w0 + self.w1 * feature
                error = y_true - y_pred
                
                self.w0 = self.w0 + learning_rate * error
                self.w1 = self.w1 + learning_rate * error * feature

                self.mse_history.append(np.mean((y - self.predict(X))**2))

        if plot_learning_curve:
            self.__plot_learning_curve()

    
    def __plot_learning_curve(self):
        
        """Plota a curva de aprendizado invertida para mostrar a parte estável na parte inferior.
        
        Args:
            mse_history (List[float]): List of MSE values recorded during training.
        """
        plt.plot(self.mse_history)
        plt.xlabel("Atualização dos pesos")
        plt.ylabel("MSE")
        plt.title("Curva de Aprendizado")  
        plt.show()
