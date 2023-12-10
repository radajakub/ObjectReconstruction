from inout import DataLoader


class Config:
    default_threshold = 3
    default_p = 0.9999
    default_max_iter = 1000

    def __init__(self, argv: list[str]) -> None:
        args = argv[1:]
        if len(args) % 2 != 0:
            print("wrong number of arguments")
            exit(1)

        self.scene = None
        self.img1 = None
        self.img2 = None
        self.outpath = None
        self.seed = None
        self.threshold = Config.default_threshold
        self.p = Config.default_p
        self.max_iter = Config.default_max_iter
        self.silent = False
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
            elif key == 'seed':
                self.seed = int(val)
            elif key == 'threshold':
                self.threshold = float(val)
            elif key == 'p':
                self.p = float(val)
            elif key == 'max_iter':
                self.max_iter = int(val)
            elif key == 'silent':
                self.silent = bool(val)

        if self.scene is None:
            raise ValueError('scene not specified')

    def check_valid(self, loader: DataLoader):
        if self.img1 is None or self.img2 is None:
            raise ValueError('img1 and img2 must be specified')
        if self.img1 > loader.image_num or self.img2 > loader.image_num or self.img1 < 1 or self.img2 < 1:
            raise ValueError(f'invalid image id -> must be between 1 and {loader.image_num} (including)')

    def __str__(self) -> str:
        print('config:')
        print(f'-- scene {self.scene} with images {self.img1} and {self.img2}')
        print(f'-- seed {self.seed}')
        print(f'-- save output image to {self.outpath}')
        print(f'-- epipolar estimation params')
        print(f'---- threshold {self.threshold}')
        print(f'---- p {self.p}')
        print(f'---- max_iter {self.max_iter}')
        print()
