from __future__ import annotations

import numpy as np

from utils import toolbox as tb


class Camera:
    @staticmethod
    def zero_camera(K: np.ndarray) -> Camera:
        return Camera(K, np.eye(3, 4), np.eye(3), np.zeros((3, 1)))

    @staticmethod
    def from_Rt(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> Camera:
        return Camera(K, np.hstack((R, t)), R, t)

    @staticmethod
    def from_P(K: np.ndarray, P: np.ndarray) -> Camera:
        Rt = np.linalg.inv(K) @ P
        return Camera(K, Rt, Rt[:, :3], Rt[:, 3][:, np.newaxis])

    @staticmethod
    def get_fundamental(P1: Camera, P2: Camera) -> np.ndarray:
        K_ = P1.K_
        R21 = P2.R @ P1.R.T
        t21 = P2.t - R21 @ P1.t
        return K_.T @ tb.sqc(-t21.squeeze()) @ R21 @ K_

    @staticmethod
    def check_visibility(P1: Camera, P2: Camera, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        X = tb.Pu2X(P1.P_Kless, P2.P_Kless, x1, x2)
        u1p = P1.project_Kless(X)
        u2p = P2.project_Kless(X)
        return np.logical_and(u1p[2, :] > 0, u2p[2, :] > 0)

    def __init__(self, K: np.ndarray, P_Kless: np.ndarray, R: np.ndarray, t: np.ndarray) -> None:
        self.K = K
        self.K_ = np.linalg.inv(K)
        self.R = R
        self.t = t
        self.P_Kless = P_Kless
        self.P = self.K @ self.P_Kless
        self.order = -1

    def set_order(self, order: int) -> None:
        self.order = order

    def decompose(self) -> tuple[np.ndarray, np.ndarray]:
        C = -self.R.T @ self.t  # camera position in world reference
        o = np.linalg.det(self.R) * self.R[2, :]  # optical axis in world reference
        return C.squeeze(), o

    def project(self, X: np.ndarray) -> np.ndarray:
        return self.P @ X

    def project_Kless(self, X: np.ndarray) -> np.ndarray:
        return self.P_Kless @ X

    def visible_indices(self, X: np.ndarray) -> np.ndarray:
        return np.arange(X.shape[1])[self.project_Kless(X)[2, :] > 0]
