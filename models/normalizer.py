import numpy as np

class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.min = None
        self.max = None
        self.feature_range = feature_range

    def fit(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        if self.min is None or self.max is None:
            raise ValueError("O normalizador precisa ser ajustado com 'fit' antes de transformar os dados.")
        scale = self.feature_range[1] - self.feature_range[0]
        return self.feature_range[0] + (data - self.min) * scale / (self.max - self.min)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data_normalized):
        if self.min is None or self.max is None:
            raise ValueError("O normalizador precisa ser ajustado com 'fit' antes de desnormalizar os dados.")
        scale = self.feature_range[1] - self.feature_range[0]
        return self.min + (data_normalized - self.feature_range[0]) * (self.max - self.min) / scale
