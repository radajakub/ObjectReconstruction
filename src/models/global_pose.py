import numpy as np

from .model import Model
from .camera import Camera
from utils import toolbox as tb
from packages import p3p


class GlobalPose(Model):
    def __init__(self, K: np.ndarray) -> None:
        super().__init__(min_samples=3)
        self.K = K
        self.K_ = np.linalg.inv(K)

    def hypothesis(self, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        u = self.unapply_K(u)
        Rts = []
        for X_ in p3p.p3p_grunert(X, u):
            Rts.append(p3p.XX2Rt_simple(X, X_))
        return Rts

    def error(self, X: np.ndarray, u: np.ndarray, c: Camera) -> np.ndarray:
        u_hat = c.project(X)
        es = []
        for u_hat_i, u_i in zip(u_hat.T, u.T):
            ei = np.array([
                u_hat_i[0] / u_hat_i[2] - u_i[0] / u_i[2],
                u_hat_i[1] / u_hat_i[2] - u_i[1] / u_i[2]
            ])
            es.append(np.dot(ei, ei))
        return np.array(es)

    def support(self, inlier_errors: np.ndarray, threshold: float) -> int:
        # here we dont raise the inlier_errors to the power of 2, they are already squared
        return np.sum(1 - inlier_errors / np.power(threshold, 2))

    def apply_K(self, X: np.ndarray) -> np.ndarray:
        return self.K @ X

    def unapply_K(self, X: np.ndarray) -> np.ndarray:
        return self.K_ @ X
