import sys
import os
import numpy as np

from inout import DataLoader, Logger, Plotter3D
from utils import Config
from estimation import PointCloud, CameraGluer

if __name__ == "__main__":
    config = Config(sys.argv)

    rng = np.random.default_rng(seed=config.seed)

    loader = DataLoader(config.scene)
    config.check_valid(loader)

    logger = Logger(config=config)
    logger.intro()

    point_cloud = PointCloud()
    camera_gluer = CameraGluer(loader, point_cloud, threshold=config.threshold, p=config.p,
                               max_iterations=config.max_iter, rng=rng, logger=logger)

    # exit(0)
    camera_gluer.initial(config.img1, config.img2)

    plotter = Plotter3D(hide_axes=True, invert_yaxis=False, aspect_equal=True)
    plotter.add_points(camera_gluer.point_cloud.get_points())
    plotter.add_camera(camera_gluer.cameras[config.img1], color='red', show_plane=True)
    plotter.add_camera(camera_gluer.cameras[config.img2], color='blue', show_plane=True)

    if config.outpath is None:
        plotter.show()
    else:
        os.makedirs(config.outpath, exist_ok=True)
        plotter.save(outfile=os.path.join(config.outpath, 'img.png'))
