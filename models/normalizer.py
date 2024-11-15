import numpy as np

class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("O normalizador precisa ser ajustado com 'fit' antes de transformar os dados.")
        return (data - self.mean) / self.std

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data_normalized):
        if self.mean is None or self.std is None:
            raise ValueError("O normalizador precisa ser ajustado com 'fit' antes de desnormalizar os dados.")
        return (data_normalized * self.std) + self.mean
