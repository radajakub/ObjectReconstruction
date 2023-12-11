import numpy as np

from packages.corresp import Corresp

from .point_cloud import PointCloud
from .ransac import RANSAC
from .epipolar import EpipolarEstimator
from utils import Config, toolbox as tb
from inout import Logger, DataLoader, CameraGluerLogEntry
from models import GlobalPose, Camera


class GlobalPoseEstimate:
    def __init__(self, R: np.ndarray, t: np.ndarray, inlier_indices: np.ndarray) -> None:
        self.R = R
        self.t = t
        self.inlier_indices = inlier_indices

    def __str__(self) -> str:
        res = 'GlobalPoseEstimate\n'
        res += 'R:\n'
        res += self.R.__str__() + '\n'
        res += 't:\n'
        res += self.t.__str__() + '\n'
        return res


class CameraGluer(RANSAC):
    def __init__(self, loader: DataLoader, point_cloud: PointCloud, threshold: float = Config.default_threshold, p: float = Config.default_p, max_iterations: int = Config.default_max_iter, rng: np.random.Generator = None, logger: Logger = None) -> None:
        super().__init__(model=GlobalPose(loader.K), threshold=threshold,
                         p=p, max_iterations=max_iterations, rng=rng)
        self.logger = logger
        self.loader = loader
        self.point_cloud = point_cloud
        self.epipolar_estimator = EpipolarEstimator(self.loader.K, threshold, p, max_iterations, rng, logger)
        self.reset_cameras()
        self.initialize_manipulator()

    def reset_cameras(self) -> None:
        self.cameras = {key: None for key in self.loader.image_ids}

    def initialize_manipulator(self) -> None:
        # initialize cameras
        self.manipulator = Corresp(self.loader.image_num)
        for i in self.loader.image_ids:
            for j in self.loader.image_ids:
                if i < j:
                    ci, cj = self.loader.get_corres(i, j)
                    mij = np.hstack((ci[:, np.newaxis], cj[:, np.newaxis]))
                    self.manipulator.add_pair(i - 1, j - 1, mij)

    def initial(self, img1: int, img2: int) -> None:
        # (1) The set of selected cameras is empty.
        self.reset_cameras()

        # (2) Choose a pair of images I1 a I2 -> done from Config

        # (3) Find the relative pose and orientation (R,t), choose (nonzero) scale, e.g., choose length of base equal to 1.
        m1, m2 = self.manipulator.get_m(img1 - 1, img2 - 1)
        corr1 = self.loader.get_points(img1, m1)
        corr2 = self.loader.get_points(img2, m2)
        estimate = self.epipolar_estimator.fit(corr1, corr2)  # in estimate there is R, t (with unit length)

        # (4) Choose the global coordinate system such that it is equal to the coordinate system of the first camera.
        # Construct the cameras P1 and P2.
        # Put these cameras into the set of selected cameras.
        P1, P2 = estimate.get_cameras()
        self.cameras[img1] = P1
        self.cameras[img2] = P2

        # (5) Reconstruct the 3D point cloud using inlier correspondences between the images I1 and I2 using the cameras P1 and P2 (the points must be in front of both cameras)
        corr_in_1, corr_in_2 = estimate.get_inliers(corr1, corr2)
        scene_indices = self.point_cloud.add_epipolar(estimate, corr_in_1, corr_in_2)

        self.manipulator.start(img1 - 1, img2 - 1, estimate.inlier_indices, scene_indices)

        # (6) Refine the camera set {P1, P2} together with the point cloud using bundle adjustment.
        # TODO

        self.logger.log(CameraGluerLogEntry([img1, img2]))

    def append_camera(self) -> None:
        # (1) Select an image (Ij) that has not a camera estimated yet
        imgs, counts = self.manipulator.get_green_cameras()
        # add 1 to get it into my indexing for consistency
        img = imgs[np.argmax(counts)] + 1

        # (2) Find image points in Ij, that correspond to some allready reconstructed 3D points in the cloud
        point_indices, corr_indices, Xu_verified = self.manipulator.get_Xu(img - 1)
        points = tb.e2p(self.point_cloud.get_points(point_indices))
        correspondences = tb.e2p(self.loader.get_points(img, corr_indices))

        # (3) Estimate the global pose and orientation of the camera Pj using the P3P algorithm in RANSAC scheme
        estimate = self.fit(points, correspondences)
        if estimate is None:
            raise Exception("Appending camera failed")

        # (5) Refine the camera Pj using numeric minimisation of reprojection errors in Ij (updates Pj only)
        # TODO

        # (4) Insert the camera Pj into the set of selected cameras
        camera = Camera.from_Rt(self.loader.K, estimate.R, estimate.t)
        self.cameras[img] = camera
        self.manipulator.join_camera(img - 1, estimate.inlier_indices)

        # (6) Find correspondences from between Ij and the images of selected cameras, that have not 3D point yet and reconstruct new 3D points and add them to the point cloud
        # for i in self.manipulator.get_cneighbours(img - 1):
        #     img_n = i + 1
        #     m_img, m_img_n = self.manipulator.get_m(img - 1, img_n - 1)
        #     print(m_img.shape, m_img_n.shape)
        #     P_i = self.cameras[img_n]

        #     # Reconstruct new scene points using the cameras i and ic and image-to-image correspondences m
        #     corr_img = tb.e2p(self.loader.get_points(img, m_img))
        #     corr_img_n = tb.e2p(self.loader.get_points(img_n, m_img_n))
        #     # TODO: some check?
        #     scene_indices = self.point_cloud.add(camera, P_i, corr_img, corr_img_n)
        #     self.manipulator.new_x(img - 1, img_n - 1, np.arange(m_img_n.shape[0]), scene_indices)

    def fit(self, scene_points: np.ndarray, image_points: np.ndarray) -> GlobalPoseEstimate:
        assert (scene_points.shape[1] == image_points.shape[1])

        # (1) Check, if there are enough correspondences. If not, terminate with no solution found.
        N = scene_points.shape[1]
        if N < self.model.min_samples:
            raise Exception("P3P: Not enough correspondences")

        # (2) Initialise number of iterations and its allowed maximum
        self.it = 0
        Nmax = np.inf

        # (3) Initialise best hypothesis and support
        best_support = 0
        best_estimate = None

        # (4) Loop while number of iterations is less than maximum allowed
        while self.it <= Nmax and self.it < self.max_iterations:
            self.it += 1
            # (5) Generate hypothesis
            # (5a) Sample 3 correspondences
            sample_idx = self.rng.choice(N, self.model.min_samples, replace=False)
            sample_scene_points = scene_points[:, sample_idx]
            sample_image_points = image_points[:, sample_idx]

            Rts = self.model.hypothesis(sample_scene_points, sample_image_points)

            for Rt in Rts:
                R, t = Rt
                # (6) Verify hypothesis
                # (6a) Compute camera
                P = Camera.from_Rt(self.model.K, R, t)

                # (6b) Select 3D points that are in front of the camera
                projections = P.project_Kless(scene_points)
                visible_indices = np.arange(projections.shape[1])[projections[2, :] > 0]
                visible_scene_points = scene_points[:, visible_indices]
                visible_image_points = image_points[:, visible_indices]
                if visible_indices.shape[0] == 0:
                    continue

                # (6c) Compute reprojection error
                err = self.model.error(visible_scene_points, visible_image_points, P)
                inliers = err < np.power(self.threshold, 2)
                inlier_indices = visible_indices[inliers]
                support = self.model.support(err[inliers], self.threshold)
                if support > best_support:
                    best_support = support
                    best_estimate = GlobalPoseEstimate(R, t, inlier_indices)
                    eps = 1 - inlier_indices.shape[0] / N
                    Nmax = 0 if eps == 0 else (np.log(1 - self.p) / np.log(1 - np.power(1 - eps, 3)))

        return best_estimate
