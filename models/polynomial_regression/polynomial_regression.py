import numpy as np
from typing import List

from models.model import Model
from models.normalizer import Normalizer


class PolynomialRegression(Model):
    """A simple linear regression model, using the Ordinary Least Squares (OLS) algorithm.
    """
    def __init__(self, polynomial_degree: int):
        self.w = []
        self.polynomial_degree = polynomial_degree
        self.scaler_X = Normalizer()
        self.scaler_y = Normalizer()
    
    def predict(self, input_array: np.ndarray, **kwargs) -> List[str]:
        """Returns the model's prediction for the given input.

        Args:
            input_array (np.ndarray): an n-dimensional numpy array to be passed to the model,\
            representing a batch of inputs.

        Returns:
            List[float]: the predicted output for each input.
        """

        if input_array.ndim == 1:
            input_array = input_array.reshape(-1, 1)

        input_array = self.scaler_X.transform(input_array)
        input_array = self.__transform_columns(input_array)

        input_array = np.c_[np.ones(input_array.shape[0]), input_array]
        predictions = input_array @ self.w 
        predictions = self.scaler_y.inverse_transform(predictions)
         
        return predictions.flatten().tolist()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Trains the model using the given training and validation data.

        Args:
            training_data (np.ndarray): the training production.
            validation_data (np.ndarray): the validation production.
            epochs (int): the number of epochs to train the model.
            learning_rate (float): the learning rate for the gradient descent algorithm.
        """
        y = y.reshape(-1,1)
        
        if X.ndim == 1:  
            X = X.reshape(-1, 1)
        
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        X = self.__transform_columns(X)

        X = np.c_[np.ones(X.shape[0]), X]

        self.w = np.linalg.pinv(X.T @ X) @ X.T @ y
        return
    
    def fit_l2(self, X: np.ndarray, y: np.ndarray, lambda_reg: float = 0.01):
        """Trains the model using L2 regularization (ridge regression).

        Args:
            X (np.ndarray): the training data inputs.
            y (np.ndarray): the training data outputs.
        """
        y = y.reshape(-1, 1)
        
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        X = self.__transform_columns(X)

        X = np.c_[np.ones(X.shape[0]), X]

        I = np.eye(X.shape[1])
        I[0, 0] = 0
        
        self.w = np.linalg.pinv(X.T @ X + lambda_reg * I) @ X.T @ y
        return

    def __transform_columns(self, X: np.ndarray):
        """Transforms the columns of the given matrix.

        Args:
            X (np.ndarray): the matrix to be transformed.

        Returns:
            np.ndarray: the transformed matrix.
        """

        transformed_columns = [X] 
        for degree in range(2, self.polynomial_degree + 1):  
            transformed_columns.append(np.power(X, degree))
        return np.hstack(transformed_columns)
        