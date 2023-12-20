import sys
import os
import numpy as np

from inout import DataLoader, Logger, Plotter, Plotter3D, ActionLogEntry
from utils import Config
from estimation import StereoMatcher

if __name__ == "__main__":
    config = Config(sys.argv)

    rng = np.random.default_rng(seed=config.seed)

    logger = Logger(config=config)
    logger.intro()

    logger.log(ActionLogEntry('Loading data'))

    loader = DataLoader(config.scene)

    stereo = StereoMatcher.load(config, loader, logger)
    stereo.load_disparities()
    stereo.fill_point_cloud()

    stereo.save()

    # plot disparities
    # (1) horizontal
    horizontal_plotter = Plotter(rows=3, cols=3)  # there is 9 of them
    for i, (pair, disparity) in enumerate(zip(*stereo.get_horizontal_disparities())):
        row = i // 3
        col = i % 3
        horizontal_plotter.set_title(f'Disparity for image pair {pair}', row=row, col=col)
        horizontal_plotter.add_image_nan(disparity, row=row, col=col)

    # (2) vertical
    vertical_plotter = Plotter(rows=2, cols=4)  # there is 8 of them
    for i, (pair, disparity) in enumerate(zip(*stereo.get_vertical_disparities())):
        row = i // 4
        col = i % 4
        vertical_plotter.set_title(f'Disparity for image pair {pair}', row=row, col=col)
        vertical_plotter.add_image_nan(disparity, row=row, col=col)

    if config.outpath is not None:
        horizontal_plotter.save(os.path.join(config.outpath, 'horizontal_disparities.png'))
        vertical_plotter.save(os.path.join(config.outpath, 'vertical_disparities.png'))
    else:
        horizontal_plotter.show()
        vertical_plotter.show()
