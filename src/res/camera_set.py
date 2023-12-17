import numpy as np
from PIL import Image

from inout import DataLoader
from .camera import Camera


class CameraSet:
    def __init__(self, loader: DataLoader) -> None:
        self.cameras = {i: None for i in loader.image_ids}
        self.images = {i: loader.images[i] for i in loader.image_ids}
        self.count = 0

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
