import numpy as np
from numpy.typing import ArrayLike
from typing.io import IO
from typing import Iterable
from tqdm.notebook import tqdm


def clamp(x, l, h):
    if x < l:
        return l
    if x > h:
        return h
    return x


class PointCloudWriter:
    def __init__(self, fp: IO, points: ArrayLike, limit: int = -1):
        self.fp = fp
        self.points = points
        self.limit = limit

    def write_header(self):
        total = self.limit if self.limit > 0 else self.points.shape[0]
        self.fp.write("ply\n")
        self.fp.write("format ascii 1.0\n")
        self.fp.write("element vertex {}\n".format(total))
        self.fp.write("property float x\n")
        self.fp.write("property float y\n")
        self.fp.write("property float z\n")
        self.fp.write("property uchar red\n")
        self.fp.write("property uchar green\n")
        self.fp.write("property uchar blue\n")
        self.fp.write("end_header\n")

    def write_body(self):
        points = tqdm(self.points)
        count = 0
        for p in points:
            x, y, z = p
            r = 255
            g = 200
            b = clamp(int(z * 255 / 2500), 0, 255)
            if z > 2500:
                continue
            count += 1
            self.fp.write(f'{x} {y} {z} {r} {g} {b}\n')
            if 0 < self.limit <= count:
                break
        points.close()

    def write(self):
        self.write_header()
        self.write_body()

    @classmethod
    def write_points(cls, fp: IO, points: ArrayLike, limit: int = -1):
        cls(fp, points, limit).write()
