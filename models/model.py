from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Model(ABC):
    """A common interface implemented by all SACI vlm. Every model in the project \
    must support the method signatures in this interface, as to guarantee interchangeability \
    and ease of use.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        """Loads a model from HuggingFace with pre-trained parameters.

        Args:
            device (str): the device where the model will be executed. "cpu" for cpu execution,\
            and "cuda:0" for gpu execution.
            endpoint (str): the endpoint from which the model will be downloaded from.
        """
        pass
    
    @abstractmethod
    def predict(self, input_array: np.ndarray, **kwargs) -> List[str]:
        """Returns the model's prediction for the given input.

        Args:
            input_array (np.ndarray): an n-dimensional numpy array to be passed to the model,\
            representing a batch of inputs.

        Returns:
            List[str]: the predicted output for each input.
        """
        pass
    
    @abstractmethod
    def fit(self, training_data: np.ndarray, validation_data: np.ndarray, **kwargs):
        """Trains the model using the given training and validation data.

        Args:
            training_data (np.ndarray): the training production.
            validation_data (np.ndarray): the validation production.
        """
        pass

