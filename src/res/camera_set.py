import os

import numpy as np
from PIL import Image

from inout import DataLoader, Logger, ActionLogEntry
from .camera import Camera
from packages.rectify import rectify


class CameraSet:
    FOLDER_NAME = 'cameras'

    def __init__(self, loader: DataLoader, logger: Logger) -> None:
        self.cameras = {i: None for i in loader.image_ids}
        self.images = {i: loader.images[i] for i in loader.image_ids}
        self.count = 0
        self.logger = logger

    def load(self, outpath: str) -> None:
        outpath = os.path.join(outpath, 'cameras')
        for folder in os.listdir(outpath):
            if os.path.isdir(os.path.join(outpath, folder)):
                camera = Camera.load(outpath, folder)
                self.cameras[camera.image] = camera
                self.count = max(self.count, camera.order)
                self.logger.log(ActionLogEntry(
                    f'Loaded camera {camera.image} with order {camera.order} from {outpath}'))

    def can_add(self) -> bool:
        return None in self.cameras.values()

    def add_camera(self, image: int, camera: Camera) -> None:
        self.count += 1
        camera.set_image_order(image, self.count)
        self.cameras[image] = camera

    def get_camera(self, image: int) -> Camera:
        return self.cameras[image]

    def get_cameras(self) -> list[Camera]:
        return sorted(self.cameras.values(), key=lambda c: c.order)

    def get_camera_image(self, image: int) -> tuple[Camera, Image.Image]:
        return self.cameras[image], self.images[image]

    def rectify(self, i1: int, i2: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, Image.Image, Image.Image]:
        P1, img1 = self.get_camera_image(i1)
        P2, img2 = self.get_camera_image(i2)
        F = Camera.get_fundamental(P1, P2)
        H1, H2, img1_r, img2_r = rectify(F, np.array(img1), np.array(img2))
        return F, H1, H2, img1_r, img2_r

    def save(self, outpath: str):
        outpath = os.path.join(outpath, CameraSet.FOLDER_NAME)
        os.makedirs(outpath, exist_ok=True)
        for camera in self.cameras.values():
            camera.save(outpath)
            self.logger.log(ActionLogEntry(f'Saved camera {camera.image} with order {camera.order} to {outpath}'))
