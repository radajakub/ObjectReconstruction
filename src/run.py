import sys
import numpy as np

from plotter import Plotter
from loader import DataLoader
from epipolar import EpipolarEstimator
from config import Config
from logger import Logger


if __name__ == "__main__":

    config = Config(sys.argv)

    rng = np.random.default_rng(seed=config.seed)

    loader = DataLoader(config.scene)
    config.check_valid(loader)

    logger = Logger(config=config, loader=loader)
    logger.intro()

    corr1, corr2 = loader.get_corresp(config.img1, config.img2)

    estimator = EpipolarEstimator(K=loader.K, threshold=config.threshold, p=config.p, max_iterations=config.max_iter, rng=rng, logger=logger)
    estimate = estimator.fit(corr1, corr2)

    E = estimate.E
    mask = np.zeros(corr1.shape[1], dtype=bool)
    mask[estimate.inlier_indices] = True
    corr_in = [corr1[:, mask], corr2[:, mask]]
    corr_out = [corr1[:, ~mask], corr2[:, ~mask]]

    images = [config.img1, config.img2]
    cols = 3
    n_epipolar_lines = 10
    n_inliers = corr_in[0].shape[1]

    plotter = Plotter(rows=len(images), cols=cols)
    for i, img in enumerate(images):
        r = i + 1

        # set titles for the plots in grid
        plotter.set_title(f'Image {img}', row=r, col=1)
        plotter.set_title(f'Inliers in image {img}', row=r, col=2)
        plotter.set_title(f'Epipolar_lines for image {img}', row=r, col=3)

        # add image to every column
        for c in range(1, cols+1):
            plotter.add_image(loader.images[img], row=r, col=c)

        # needle plot of inliers and outliers of correspondences
        j = (i + 1) % 2
        plotter.add_needles(corr_out[j], corr_out[i], color='black', row=i, col=2)
        plotter.add_needles(corr_in[j], corr_in[i], color='red', row=i, col=2)

    # plot epipolar lines and points
    for c, k in enumerate(range(0, n_inliers, n_inliers // n_epipolar_lines)):
        ls = estimator.compute_epipolar_lines(estimate, corr_in[0][:, k], corr_in[1][:, k])
        color = plotter.get_color(c)
        for i, l in enumerate(ls):
            r = i + 1
            plotter.add_point(corr_in[i][:, k], color=color, size=10, row=r, col=3)
            plotter.add_line(l, color=color, linewidth=1, row=r, col=3)

    logger.dump()
    logger.summary()
    logger.outro()

    if config.outpath is None:
        plotter.show()
    else:
        plotter.save(outfile=config.outpath)
