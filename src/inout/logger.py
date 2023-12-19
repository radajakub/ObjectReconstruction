import time
import os

from utils import Config


class LogEntry:
    def __init__(self, label: str) -> None:
        self.time = 0
        self.label = label

    def __str__(self) -> str:
        return self.label

    def add_timestamp(self, start_time: float) -> None:
        self.time = round(time.process_time() - start_time, 4)


class ActionLogEntry(LogEntry):
    def __init__(self, action: str) -> None:
        super().__init__(label='Action')
        self.action = action

    def __str__(self) -> str:
        return f'(Action) [{self.time}] | {self.action}'


class EpipolarEstimateLogEntry(LogEntry):
    def __init__(self, iteration: int, inliers: int, support: float, visible: int,  Nmax: float) -> None:
        super().__init__(label='EpipolarEstimate')
        self.iteration = iteration
        self.inliers = inliers
        self.support = support
        self.visible = visible
        self.Nmax = Nmax

    def __str__(self) -> None:
        return f'({self.label}) [{self.time}] | it: {self.iteration}] inliers: {self.inliers}, support: {self.support}, visible: {"-" if self.visible == 0 else self.visible}, Nmax: {self.Nmax}'


class RANSACLogEntry(LogEntry):
    def __init__(self, iteration: int, inliers: int, support: float, Nmax: float) -> None:
        super().__init__(label='RANSAC')
        self.iteration = iteration
        self.inliers = inliers
        self.support = support
        self.Nmax = Nmax

    def __str__(self) -> str:
        return f'({self.label}) [{self.time}] | it: {self.iteration}] inliers: {self.inliers}, support: {self.support}, Nmax: {self.Nmax}'


class CameraGluerLogEntry(LogEntry):
    def __init__(self, points: int, P1: int, P2: int = None) -> None:
        super().__init__(label='CameraGluer')
        self.P1 = P1
        self.P2 = P2
        self.points = points

    def __str__(self) -> str:
        if self.P2 is not None:
            return f'({self.label}) [{self.time}] initialized with cameras {self.P1} and {self.P2}, scene points : {self.points}'
        return f'({self.label}) [{self.time}] glued camera {self.P1}, scene points : {self.points}'


class GlobalPoseLogEntry(LogEntry):
    def __init__(self, iteration: int, inliers: int, visible: int, support: float, Nmax: float) -> None:
        super().__init__(label='GlobalPose')
        self.iteration = iteration
        self.inliers = inliers
        self.visible = visible
        self.support = support
        self.Nmax = Nmax

    def __str__(self) -> str:
        return f'({self.label}) [{self.time}] | it: {self.iteration}] inliers: {self.inliers}, support: {self.support}, visible: {"-" if self.visible == 0 else self.visible}, Nmax: {self.Nmax}'


class Logger:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.log_clean()

    def log_clean(self) -> None:
        self.start_time = time.process_time()
        self.logs = []

    def log(self, entry: LogEntry) -> None:
        entry.add_timestamp(start_time=self.start_time)
        self.logs.append(entry)
        if not self.config.silent:
            print(entry)

    def intro(self) -> None:
        if not self.config.silent:
            print(self.config)

    def dump(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'log.txt'), 'w') as f:
            for log in self.logs:
                f.write(f'{str(log)} + \n')
