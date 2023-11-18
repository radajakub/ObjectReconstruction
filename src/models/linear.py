import numpy as np
import toolbox as tb

from models.model import Model


class Linear(Model):
    def __init__(self) -> None:
        super().__init__(min_samples=2)

    def hypothesis(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[1]
        mu = (np.sum(X, axis=1) / m)[:, np.newaxis]
        At = X - mu
        values, vectors = np.linalg.eig(At @ At.T)
        n = vectors[:, np.argmin(values)]
        # normalise n
        n = n / np.linalg.norm(n)
        d = np.dot(-n.T, mu)
        return np.hstack((n, d))[:, np.newaxis]

    def error(self, hypothesis: np.ndarray, X: np.ndarray) -> np.ndarray:
        numerator = hypothesis.T.dot(tb.e2p(X)).squeeze()
        denominator = np.sqrt(np.sum(np.power(hypothesis[0:2], 2)))
        return np.abs(numerator / denominator)

    def support(self, inlier_errors: np.ndarray) -> int:
        return inlier_errors.shape[0]
