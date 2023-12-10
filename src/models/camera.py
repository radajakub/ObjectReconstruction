import numpy as np

from .model import Model
from utils import toolbox as tb


class Camera(Model):
    def __init__(self, K: np.ndarray, P: np.ndarray = None) -> None:
        super().__init__(min_samples=3)
        self.P = P
        self.K = K
        self.K_ = np.linalg.inv(K)

    def hypothesis(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        assert (X1.shape == X2.shape)

    def error(self, E: np.ndarray, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return np.sqrt(tb.err_F_sampson(self.get_fundamental(E), self.apply_K(X1), self.apply_K(X2)))

    def support(self, inlier_errors: np.ndarray, threshold: float) -> int:
        return np.sum(1 - np.power(inlier_errors, 2) / np.power(threshold, 2))

    def decompose(self) -> tuple[np.ndarray, np.ndarray]:
        Rt = self.unapply_K(self.P)
        R, t = Rt[:, :3], Rt[:, 3]
        C = -R.T @ t  # camera position in world reference
        o = np.linalg.det(R) * R[2, :]  # optical axis in world reference
        return C, o

    def apply_K(self, X: np.ndarray) -> np.ndarray:
        return self.K @ X

    def unapply_K(self, X: np.ndarray) -> np.ndarray:
        return self.K_ @ X
