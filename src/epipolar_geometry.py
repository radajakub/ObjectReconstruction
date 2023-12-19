import sys
import os
import numpy as np

from inout import Plotter, DataLoader, Logger, ActionLogEntry
from inout import DataLoader
from estimation import EpipolarEstimator
from utils import Config


if __name__ == "__main__":
    config = Config(sys.argv)

    logger = Logger(config=config)
    logger.intro()

    rng = np.random.default_rng(seed=config.seed)

    logger.log(ActionLogEntry('Loading data'))
    loader = DataLoader(config.scene)
    config.check_valid(loader)

    corr1, corr2 = loader.get_corresp(config.img1, config.img2)

    estimator = EpipolarEstimator(K=loader.K, config=config, rng=rng, logger=logger)
    estimate = estimator.fit(corr1, corr2)

    corr_in = estimate.get_inliers(corr1, corr2)
    corr_out = estimate.get_outliers(corr1, corr2)

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

    if config.outpath is None:
        plotter.show()
    else:
        os.makedirs(config.outpath, exist_ok=True)
        logger.dump(config.outpath)
        plotter.save(outfile=os.path.join(config.outpath, f'epipolar_{config.img1}_{config.img2}.png'))
