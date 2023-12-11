from __future__ import annotations

import numpy as np


class Camera:
    @staticmethod
    def zero_camera(K: np.ndarray) -> Camera:
        return Camera(K, np.eye(3, 4), np.eye(3), np.zeros((3, 1)))

    @staticmethod
    def from_Rt(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> Camera:
        return Camera(K, np.hstack((R, t)), R, t)

    def __init__(self, K: np.ndarray, P: np.ndarray, R: np.ndarray, t: np.ndarray) -> None:
        self.K = K
        self.K_ = np.linalg.inv(K)
        self.R = R
        self.t = t
        self.P_Kless = P
        self.P = self.K @ self.P_Kless

    def decompose(self) -> tuple[np.ndarray, np.ndarray]:
        C = -self.R.T @ self.t  # camera position in world reference
        o = np.linalg.det(self.R) * self.R[2, :]  # optical axis in world reference
        return C.squeeze(), o

    def project(self, X: np.ndarray) -> np.ndarray:
        return self.P @ X

    def project_Kless(self, X: np.ndarray) -> np.ndarray:
        return self.P_Kless @ X
