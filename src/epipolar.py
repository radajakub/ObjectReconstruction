import numpy as np
from loader import DataLoader


class Hypothesis:
    def __init__(self, X1: np.ndarray, X2: np.ndarray, sample_size: int = 5) -> None:
        assert (X1.shape == X2.shape)
        sample_indices = np.random.choice(X1.shape[1], sample_size, replace=False)
        self.X1 = X1[:, sample_indices]
        self.X2 = X2[:, sample_indices]

        self.Es = [E for E in self._find_Es() if self._check(E)]

    def _find_Es(self):
        return [np.eye(3)]

    def _check(self, E: np.ndarray) -> bool:
        return True


class EpiploarEstimator:
    def __init__(self, data: DataLoader, img1: int, img2: int) -> None:
        # precompute inverse calibration
        self.K = data.K
        self.K_ = np.linalg.inv(self.K)
        # obtain image correspondences and unapply calibration matrix K
        self.X1, self.X2 = data.get_corresp(img1, img2)
        self.X1 = self.unapply_K(self.X1)
        self.X2 = self.unapply_K(self.X2)

    def unapply_K(self, points: np.ndarray) -> np.ndarray:
        return self.K_ @ points

    def apply_K(self, points: np.ndarray) -> np.ndarray:
        return self.K @ points

    def estimate(self):
        hypothesis = Hypothesis(self.X1, self.X2)
