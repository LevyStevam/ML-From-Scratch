from ..model import Model
from ..normalizer import Normalizer
import numpy as np

class GaussianDiscriminantAnalysis(Model):
    def __init__(self):
        self.scaler = Normalizer()
        self.mean_c1 = None
        self.mean_c2 = None
        self.covariance_c1 = None
        self.covariance_c2 = None
        self.prior_c1 = None
        self.prior_c2 = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Calcula as médias e matrizes de covariância para cada classe.
        :param X: np.ndarray, características (n_samples, n_features)
        :param y: np.ndarray, rótulos (n_samples,)
        """
        X = self.scaler.fit_transform(X)
        X_c1 = X[y == 0]
        X_c2 = X[y == 1]
      
        self.mean_c1 = np.mean(X_c1, axis=0)
        self.mean_c2 = np.mean(X_c2, axis=0)
      
        self.covariance_c1 = np.cov(X_c1, rowvar=False)
        self.covariance_c2 = np.cov(X_c2, rowvar=False)

        self.prior_c1 = X_c1.shape[0] / X.shape[0]
        self.prior_c2 = X_c2.shape[0] / X.shape[0]
    
    def predict(self, X: np.ndarray):
        """Classifica as amostras."""
        X = self.scaler.transform(X)
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray):
        """Calcula p(C_k | x) para cada classe."""
        probs = []
        for x in X:
            p_x_c1 = self.multivariate_gaussian(x, self.mean_c1, self.covariance_c1)
            p_x_c2 = self.multivariate_gaussian(x, self.mean_c2, self.covariance_c2)
            
            posterior_c1 = p_x_c1 * self.prior_c1
            posterior_c2 = p_x_c2 * self.prior_c2
            
            total = posterior_c1 + posterior_c2

            if total > 0:
                prob_c1 = posterior_c1 / total
                prob_c2 = posterior_c2 / total
            else:
                prob_c1 = prob_c2 = 0
            
            probs.append([prob_c1, prob_c2])
        return np.array(probs)

    def multivariate_gaussian(self, x, mean, covariance):
        """Calcula a densidade Gaussiana multivariada."""
        d = len(x) 
        det = np.linalg.det(covariance) 
        inv = np.linalg.inv(covariance) 
        term1 = 1 / np.sqrt((2 * np.pi) ** d * det) 
        term2 = np.exp(-0.5 * (x - mean).T @ inv @ (x - mean)) 
        return term1 * term2