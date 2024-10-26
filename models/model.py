from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Model(ABC):
    """A common interface for the models in the project.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        """Initializes the model with the given parameters.
        """
        pass
    
    @abstractmethod
    def predict(self, input_array: np.ndarray, **kwargs) -> List[str]:
        """Returns the model's prediction for the given input.

        Args:
            input_array (np.ndarray): an n-dimensional numpy array to be passed to the model,\
            representing a batch of inputs.

        Returns:
            List[float]: the predicted output for each input.
        """
        pass
    
    @abstractmethod
    def fit(self, training_data: np.ndarray, validation_data: np.ndarray, **kwargs):
        """Trains the model using the given training.

        Args:
            training_data (np.ndarray): the training production.
        """
        pass

