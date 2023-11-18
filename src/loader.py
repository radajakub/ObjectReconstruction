import numpy as np
import os
from PIL import Image

BASE_PATH = 'data'
IMAGE_FOLDER = 'images'
CORRESPONDENCE_FOLDER = 'corresp'


def two_digit_str(num: int) -> str:
    if num < 10:
        return f'0{num}'
    return str(num)


def _image_key(img1: int, img2: int) -> (int, int):
    img1, img2 = sorted([img1, img2])
    return (img1, img2)


def _get_calibration(path: str) -> np.ndarray:
    return np.loadtxt(os.path.join(path, 'K.txt'))


def _get_correspondences(path: str, img1: int, img2: int) -> np.ndarray:
    id1, id2 = _image_key(img1, img2)
    return np.loadtxt(os.path.join(path, CORRESPONDENCE_FOLDER, f'm_{two_digit_str(id1)}_{two_digit_str(id2)}.txt'), dtype=int).T


def _get_images(path: str) -> dict:
    path = os.path.join(path, IMAGE_FOLDER)
    images = {}
    for imfile in os.listdir(path):
        if imfile.endswith('.jpg'):
            id = int(os.path.splitext(os.path.basename(imfile))[0])
            if id not in images:
                images[id] = Image.open(os.path.join(path, imfile))
    return images


def _get_image_points(path: str, img: int) -> str:
    return np.loadtxt(os.path.join(path, CORRESPONDENCE_FOLDER, f'u_{two_digit_str(img)}.txt')).T


class DataLoader:
    def __init__(self, scene: str) -> None:
        self.scene = scene
        path = os.path.join(BASE_PATH, scene)

        # load K matrix
        self.K = _get_calibration(path)

        # load images
        self.images = _get_images(path)
        self.image_ids = sorted(self.images.keys())
        self.image_num = len(self.image_ids)

        # load data points of all images
        self.points = {}
        for img in self.image_ids:
            self.points[img] = _get_image_points(path, img)

        # load correspondences and filter data points for each pair of images
        self.corresp = {}
        for i in range(self.image_num):
            for j in range(i + 1, self.image_num):
                key = _image_key(self.image_ids[i], self.image_ids[j])
                img1, img2 = key
                if img1 != img2 and key not in self.corresp:
                    correspondences = _get_correspondences(path, img1, img2)
                    d1 = self.points[img1][:, correspondences[0, :]]
                    d2 = self.points[img2][:, correspondences[1, :]]
                    self.corresp[key] = (d1, d2)

    def __str__(self) -> str:
        s = f'Data of scene: {self.scene}\n'
        s += f'- K:\n{self.K}\n'
        s += f'- images: {self.image_ids}\n'
        s += f'- points:\n'
        for img in self.image_ids:
            s += f'-- {img}: {self.points[img].shape}\n'
        s += f'- correspondences:\n'
        for key in self.corresp:
            s += f'-- {key}: {self.corresp[key][0].shape}\n'
        return s

    # return correponding data points for given pair of images (img1, img2) in given order
    def get_corresp(self, img1: int, img2: int) -> (np.ndarray, np.ndarray):
        key = _image_key(img1, img2)
        c1, c2 = self.corresp[key]
        return (c1, c2) if (img1, img2) == key else (c2, c1)
