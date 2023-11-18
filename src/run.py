import sys
import numpy as np

from plotter import Plotter
from loader import DataLoader
from epipolar import EpipolarEstimator
import toolbox as tb


class Config:
    def __init__(self, argv: list[str]) -> None:
        args = argv[1:]
        if len(args) % 2 != 0:
            print("wrong number of arguments")
            exit(1)

        self.scene = None
        self.img1 = None
        self.img2 = None
        for i in range(0, len(args), 2):
            key, val = args[i], args[i+1]
            if key == '--scene':
                self.scene = val
            elif key == '--img1':
                self.img1 = int(val)
            elif key == '--img2':
                self.img2 = int(val)

        if self.scene is None:
            raise ValueError('scene not specified')

        if self.img1 is None or self.img2 is None:
            raise ValueError('one of the images images is not specified, nothing is shown')

    def check_valid(self, loader: DataLoader):
        if self.img1 > loader.image_num or config.img2 > loader.image_num or config.img1 < 1 or config.img2 < 1:
            raise ValueError(f'invalid image id -> must be between 1 and {loader.image_num} (including)')


if __name__ == "__main__":
    rng = np.random.default_rng()

    config = Config(sys.argv)

    loader = DataLoader(config.scene)
    config.check_valid(loader)
    # print(loader)

    corr1, corr2 = loader.get_corresp(config.img1, config.img2)

    estimator = EpipolarEstimator(K=loader.K, threshold=5, p=0.999, max_iterations=1000)
    estimator.fit(corr1, corr2)

    E = estimator.estimate
    corr_in = [corr1[:, estimator.inliers], corr2[:, estimator.inliers]]
    corr_out = [corr1[:, ~estimator.inliers], corr2[:, ~estimator.inliers]]

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
        ls = estimator.compute_epipolar_lines(E, corr_in[0][:, k], corr_in[1][:, k])
        color = plotter.get_color(c)
        for i, l in enumerate(ls):
            r = i + 1
            plotter.add_point(corr_in[i][:, k], color=color, size=10, row=r, col=3)
            plotter.add_line(l, color=color, linewidth=1, row=r, col=3)

    plotter.show()
