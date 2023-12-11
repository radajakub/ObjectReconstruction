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

    point_cloud = PointCloud(loader.K)
    camera_gluer = CameraGluer(loader, point_cloud, threshold=config.threshold, p=config.p,
                               max_iterations=config.max_iter, rng=rng, logger=logger)

    camera_gluer.initial(config.img1, config.img2)

    print(f'{len(camera_gluer.get_cameras())} cameras: {point_cloud.get_size()}')

    while None in camera_gluer.cameras.values():
        camera_gluer.append_camera()
        print(f'{len(camera_gluer.get_cameras())} cameras: {point_cloud.get_size()}')

    plotter = Plotter3D(hide_axes=True, invert_yaxis=False, aspect_equal=True)
    plotter.add_cameras(camera_gluer.get_cameras())

    if config.outpath is None:
        plotter.show()
    else:
        os.makedirs(config.outpath, exist_ok=True)
        plotter.save(outfile=os.path.join(config.outpath, 'img.png'))
