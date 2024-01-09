import os
from typing import Optional

from inout import DataLoader


class Config:
    true_vals = ['true', 'True', 1, '1']

    default_threshold = 3
    default_p = 0.9999
    default_max_iter = 1000
    default_pose_threshold = 3
    default_reprojection_threshold = 3
    default_fundamental_threshold = 0.05

    def __init__(self, argv: Optional[list[str]]) -> None:
        self.scene = None
        self.img1 = None
        self.img2 = None
        self.outpath = None
        self.inpath = None
        self.seed = None
        self.threshold = Config.default_threshold
        self.p = Config.default_p
        self.max_iter = Config.default_max_iter
        self.silent = False
        self.pose_threshold = Config.default_threshold
        self.reprojection_threshold = Config.default_reprojection_threshold
        self.fundamental_threshold = Config.default_fundamental_threshold

        if argv is None:
            return

        args = argv[1:]
        if len(args) % 2 != 0:
            print("wrong number of arguments")
            exit(1)

        for i in range(0, len(args), 2):
            key, val = args[i], args[i+1]
            key = key.strip('-')
            if key == 'scene':
                self.scene = val
            elif key == 'img1':
                self.img1 = int(val)
            elif key == 'img2':
                self.img2 = int(val)
            elif key == 'out':
                self.outpath = val
            elif key == 'in':
                self.inpath = val
            elif key == 'seed':
                self.seed = int(val)
            elif key == 'threshold':
                self.threshold = float(val)
            elif key == 'pose-threshold':
                self.pose_threshold = float(val)
            elif key == 'reprojection-threshold':
                self.reprojection_threshold = float(val)
            elif key == 'fundamental-threshold':
                self.fundamental_threshold = float(val)
            elif key == 'p':
                self.p = float(val)
            elif key == 'max_iter':
                self.max_iter = int(val)
            elif key == 'silent':
                self.silent = bool(val)

        if self.scene is None:
            raise ValueError('scene not specified')

        if self.outpath is not None:
            self.outpath = os.path.join(self.outpath, self.scene)

        if self.inpath is not None:
            self.inpath = os.path.join(self.inpath, self.scene)

    def check_images_correct(self, loader: DataLoader):
        if self.img1 > loader.image_num or self.img2 > loader.image_num or self.img1 < 1 or self.img2 < 1:
            raise ValueError(f'invalid image id -> must be between 1 and {loader.image_num} (including)')

    def check_images_given(self, loader: DataLoader):
        if self.img1 is None or self.img2 is None:
            raise ValueError('img1 and img2 must be specified for epipolar estimation')

    def __str__(self) -> str:
        res = 'config:\n'
        res += f'-- scene {self.scene} with images {self.img1} and {self.img2}\n'
        res += f'-- seed {self.seed}\n'
        res += f'-- save output image to {self.outpath}\n'
        res += f'-- epipolar estimation params\n'
        res += f'---- threshold {self.threshold}\n'
        res += f'---- p {self.p}\n'
        res += f'---- max_iter {self.max_iter}\n'
        res += f'-- pose estimation params\n'
        res += f'---- threshold {self.pose_threshold}\n'
        res += f'-- reprojection params\n'
        res += f'---- threshold {self.reprojection_threshold}\n'
        res += f'-- stereo matching params\n'
        res += f'---- threshold {self.fundamental_threshold}\n'
        return res
