import numpy as np
from collections import Counter

from ..model import Model

class Knn:
    def __init__(self, k=3):
        """
        Inicializa o modelo k-NN.
        :param k: Número de vizinhos mais próximos a considerar.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Ajusta o modelo armazenando os dados de treinamento.
        :param X: np.ndarray, características (n_samples, n_features)
        :param y: np.ndarray, rótulos (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray):
        """
        Prediz os rótulos para as amostras fornecidas.
        :param X: np.ndarray, características das amostras de teste (n_samples, n_features)
        :return: np.ndarray, rótulos preditos (n_samples,)
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x: np.ndarray):
        """
        Prediz o rótulo de uma única amostra.
        :param x: np.ndarray, características da amostra (n_features,)
        :return: Rótulo predito
        """
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_neighbors = self.y_train[k_indices]
        most_common = Counter(k_neighbors).most_common(1)
        return most_common[0][0]

    