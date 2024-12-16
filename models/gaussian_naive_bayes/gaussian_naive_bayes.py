import numpy as np

from ..model import Model

class GaussianNaiveBayes(Model):
    def __init__(self):
        self.classes = None  
        self.means = None    
        self.variances = None  
        self.priors = None   

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Ajusta o modelo aos dados.
        :param X: np.ndarray, características (n_samples, n_features)
        :param y: np.ndarray, rótulos (n_samples,)
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
       
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.means[idx, :] = np.mean(X_cls, axis=0)
            self.variances[idx, :] = np.var(X_cls, axis=0)
            self.priors[idx] = X_cls.shape[0] / X.shape[0] 

    def log_scores(self, X: np.ndarray):
        """
        Calcula os log-scores para cada classe.
        :param X: np.ndarray, características (n_samples, n_features)
        :return: np.ndarray, log-scores de cada classe (n_samples, n_classes)
        """
        n_samples, _ = X.shape
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes))

        for idx, _ in enumerate(self.classes):
            mean = self.means[idx, :]
            variance = self.variances[idx, :]
            prior = np.log(self.priors[idx])

            term1 = -0.5 * np.sum(np.log(2 * np.pi * variance))
            term2 = -0.5 * np.sum(((X - mean) ** 2) / variance, axis=1)
            log_probs[:, idx] = prior + term1 + term2

        return log_probs

    def predict(self, X: np.ndarray):
        """
        Classifica as amostras usando diretamente o argmax nos log-scores.
        :param X: np.ndarray, características (n_samples, n_features)
        :return: np.ndarray, rótulos preditos (n_samples,)
        """
        log_probs = self.log_scores(X)
        return np.argmax(log_probs, axis=1)