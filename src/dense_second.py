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

    logger.log(ActionLogEntry('Save rectified images and corresponding disparity maps'))

    disparity_path = os.path.join(config.outpath, 'disparities')
    os.makedirs(disparity_path, exist_ok=True)

    # plot disparities
    for disps in [stereo.get_horizontal_disparities(), stereo.get_vertical_disparities()]:
        for pair, disparity in zip(*disps):
            i1, i2 = pair
            img1, img2 = stereo.get_rectified_to_plot(i1, i2)
            plotter = Plotter(rows=1, cols=3)
            plotter.set_title(f'Image {i1}', row=1, col=1)
            plotter.add_image(img1, row=1, col=1)
            plotter.set_title(f'Image {i2}', row=1, col=2)
            plotter.add_image(img2, row=1, col=2)
            plotter.set_title(f'Disparity map', row=1, col=3)
            plotter.add_image_nan(disparity, row=1, col=3)
            if config.outpath is not None:
                plotter.save(os.path.join(disparity_path, f'{i1}_{i2}.png'))
            else:
                plotter.show()

    logger.log(ActionLogEntry('FINISHED'))
