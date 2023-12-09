import sys
import os
import numpy as np

from inout import DataLoader, Logger, Plotter3D
from utils import Config
from result import PointCloud
from estimation import EpipolarEstimator

if __name__ == "__main__":
    config = Config(sys.argv)

    rng = np.random.default_rng(seed=config.seed)

    loader = DataLoader(config.scene)
    config.check_valid(loader)

    logger = Logger(config=config)
    logger.intro()

    point_cloud = PointCloud()

    corr1, corr2 = loader.get_corresp(config.img1, config.img2)

    estimator = EpipolarEstimator(K=loader.K, threshold=config.threshold, p=config.p,
                                  max_iterations=config.max_iter, rng=rng, logger=logger)
    estimate = estimator.fit(corr1, corr2)

    corr_in = estimate.get_inliers(corr1, corr2)
    point_cloud.add(estimate, corr_in[0], corr_in[1])

    plotter = Plotter3D()
    plotter.add_points(point_cloud.get_points())

    logger.dump()
    logger.summary()
    logger.outro()

    if config.outpath is None:
        plotter.show()
    else:
        os.makedirs(config.outpath, exist_ok=True)
        plotter.save(outfile=os.path.join(config.outpath, 'img.png'))
