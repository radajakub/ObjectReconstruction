from __future__ import annotations
from typing import Callable

import numpy as np
import scipy.optimize as opt
from utils import toolbox as tb


class Camera:
    @staticmethod
    def reprojection_error(u: np.ndarray, u_hat: np.ndarray) -> np.ndarray:
        es = []
        for u_hat_i, u_i in zip(u_hat.T, u.T):
            ei = np.array([
                u_hat_i[0] / u_hat_i[2] - u_i[0] / u_i[2],
                u_hat_i[1] / u_hat_i[2] - u_i[1] / u_i[2]
            ])
            es.append(np.dot(ei, ei))
        return np.array(es)

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

    @staticmethod
    def rodriguez(params: np.ndarray) -> np.ndarray:
        return Camera._rodriguez(params[0], params[1], params[2])

    @staticmethod
    def _rodriguez(phi: float, a1: float, a2: float) -> np.ndarray:
        psi = tb.sqc(phi * np.array([a1, a2, 1 - a1 - a2]))
        return np.eye(3) + (np.sin(phi) / phi) * psi + ((1 - np.cos(phi)) / (phi ** 2)) * psi @ psi

    @staticmethod
    def _opt_P(params: np.ndarray, P: Camera, scene_points: np.ndarray, err_fun: Callable[[np.ndarray], float]) -> float:
        NP = Camera.from_params(P, params)
        reprojected = NP.project(scene_points)
        err = err_fun(reprojected)
        return err

    @staticmethod
    def from_params(P: Camera, params: np.ndarray) -> Camera:
        R = Camera.rodriguez(params[:3])
        t = params[3:][:, np.newaxis]
        return Camera.from_Rt(P.K, R @ P.R, t + P.t)

    @staticmethod
    def _opt_Ps(params: np.ndarray, P: Camera, err_fun: Callable[[Camera], float]) -> float:
        return err_fun(Camera.from_params(P, params))

    @staticmethod
    def refine_pair(P1: Camera, P2: Camera, corr1: np.ndarray, corr2: np.ndarray) -> tuple[Camera, Camera]:
        x0 = np.array([1, 0, 0, 0, 0, 0])
        def err(P): return np.sum(tb.err_F_sampson(Camera.get_fundamental(P1, P), corr1, corr2))
        res = opt.fmin(Camera._opt_Ps, x0=x0, args=(P2, err))
        return P1, Camera.from_params(P2, res)

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

    def refine(self, scene_inliers: np.ndarray, image_inliers: np.ndarray) -> Camera:
        x0 = np.array([1, 0, 0, 0, 0, 0])
        def err(reprojected): return np.sum(Camera.reprojection_error(image_inliers, reprojected))
        res = opt.fmin(Camera._opt_P, x0=x0, args=(self, scene_inliers, err))
        return Camera.from_params(self, res)
