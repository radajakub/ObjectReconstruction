import numpy as np
from typing import Callable


class Model:
    def __init__(self, min_samples: int) -> None:
        self.min_samples = min_samples

    def hypothesis(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def error(self, hypothesis: np.ndarray, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def support(self, inlier_errors: np.ndarray) -> int:
        raise NotImplementedError

    def to_callable(self, params: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        raise NotImplementedError
