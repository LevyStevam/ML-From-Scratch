from models.model import Model
import numpy as np

class LinearRegression(Model):
    """A simple linear regression model.
    """
    def __init__(self, device: str, endpoint: str):
        pass
    
    def predict(self, input_array: np.ndarray, **kwargs) -> List[str]:
        """Returns the model's prediction for the given input.

        Args:
            input_array (np.ndarray): an n-dimensional numpy array to be passed to the model,\
            representing a batch of inputs.

        Returns:
            List[str]: the predicted output for each input.
        """
        pass
    
    def fit(self, training_data: np.ndarray, validation_data: np.ndarray, **kwargs):
        """Trains the model using the given training and validation data.

        Args:
            training_data (np.ndarray): the training production.
            validation_data (np.ndarray): the validation production.
        """
        pass