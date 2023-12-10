import numpy as np

from .point_cloud import PointCloud
from .ransac import RANSAC
from .epipolar import EpipolarEstimator
from utils import Config
from inout import Logger, DataLoader, CameraGluerLogEntry
from models import Camera


class CameraGluer(RANSAC):
    def __init__(self, loader: DataLoader, point_cloud: PointCloud, threshold: float = Config.default_threshold, p: float = Config.default_p, max_iterations: int = Config.default_max_iter, rng: np.random.Generator = None, logger: Logger = None) -> None:
        super().__init__(model=Camera(loader.K), threshold=threshold,
                         p=p, max_iterations=max_iterations, rng=rng)
        self.logger = logger
        self.loader = loader
        self.point_cloud = point_cloud
        self.epipolar_estimator = EpipolarEstimator(self.loader.K, threshold, p, max_iterations, rng, logger)
        self.reset_cameras()

    def reset_cameras(self) -> None:
        self.cameras = {key: None for key in self.loader.image_ids}

    def initial(self, img1: int, img2: int) -> None:
        # (1) The set of selected cameras is empty.
        self.reset_cameras()

        # (2) Choose a pair of images I1 a I2 -> done from Config

        # (3) Find the relative pose and orientation (R,t), choose (nonzero) scale, e.g., choose length of base equal to 1.
        corr1, corr2 = self.loader.get_corresp(img1, img2)
        estimate = self.epipolar_estimator.fit(corr1, corr2)  # in estimate there is R, t (with unit length)

        # (4) Choose the global coordinate system such that it is equal to the coordinate system of the first camera.
        # Construct the cameras P1 and P2.
        # Put these cameras into the set of selected cameras.
        P1, P2 = estimate.get_cameras()
        self.cameras[img1] = Camera(self.loader.K, P1)
        self.cameras[img2] = Camera(self.loader.K, P2)

        # (5) Reconstruct the 3D point cloud using inlier correspondences between the images I1 and I2 using the cameras P1 and P2 (the points must be in front of both cameras)
        corr_in_1, corr_in_2 = estimate.get_inliers(corr1, corr2)
        self.point_cloud.add(estimate, corr_in_1, corr_in_2)

        # (6) Refine the camera set {P1, P2} together with the point cloud using bundle adjustment.

        self.logger.log(CameraGluerLogEntry([img1, img2]))
