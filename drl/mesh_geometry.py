"""按夹指分层求取碰撞代理的包围盒。"""
import numpy as np

def slab_bounds(triangles, lo, hi):
    """三角面与一个 z 分层相交部分的包络，包含跨层边与层面的交点。

    不能只挑原网格中位于该层的顶点：一个很长的三角面可能跨过整层。
    """
    vertices = triangles.reshape(-1, 3)
    points = [vertices[(vertices[:, 2] >= lo) & (vertices[:, 2] <= hi)]]
    edges = triangles[:, [[0, 1], [1, 2], [2, 0]], :].reshape(-1, 2, 3)
    direction = edges[:, 1] - edges[:, 0]
    moving = direction[:, 2] != 0
    starts, directions = edges[moving, 0], direction[moving]
    for plane in (lo, hi):
        t = (plane - starts[:, 2]) / directions[:, 2]
        inside = (t >= 0) & (t <= 1)
        points.append(starts[inside] + t[inside, None] * directions[inside])
    points = np.concatenate(points)
    if not len(points):
        return None
    return np.array([points.min(axis=0), points.max(axis=0)])
