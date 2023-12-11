from estimation import PointCloud, Camera
from packages import ge as ge


def ply_export(point_cloud: PointCloud, cameras: list[Camera], path: str = 'out') -> None:
    g = ge.GePly(path + '.ply')
    g.points(point_cloud.get_all())
    g.close()
