from typing import Tuple, Union
import numpy as np
from numpy.linalg import inv as inverse_matrix

Number = Union[int, float]
Point2d = Tuple[Number, Number]


def intrinsics_matrix(focus: Point2d, center: Point2d) -> np.ndarray:
    fx, fy = focus
    cx, cy = center
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])


def pixel_to_camera_matrix(focus: Point2d, center: Point2d) -> np.ndarray:
    return inverse_matrix(intrinsics_matrix(focus, center))


default_focus: Point2d = (585, 585)
default_center: Point2d = (320, 240)

# camera to pixel
default_intrinsics_matrix = intrinsics_matrix(default_focus, default_center)

pixel_to_camera = inverse_matrix(default_intrinsics_matrix)


def scale_focus(focus: Point2d, scale_ratio: Point2d) -> Point2d:
    fx, fy = focus
    a, b = scale_ratio
    return fx * a, fy * b
