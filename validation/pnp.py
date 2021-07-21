import cv2 as cv
import numpy as np
from data.worldpos import product2
from .projection import default_intrinsics_matrix
from transforms3d import quaternions as tsq
from scipy.spatial.transform.rotation import Rotation


def solve_pnp(world_pos: np.ndarray, pixel_pos: np.ndarray, intrinsics_mat: np.ndarray):
    _retval, rvecs, tvecs, _inliers = cv.solvePnPRansac(world_pos, pixel_pos, intrinsics_mat, np.zeros(5))
    return rvecs, tvecs


def camera_pos_from_output(mat: np.ndarray, intrinsics_mat: np.ndarray = default_intrinsics_matrix):
    h, w, *_channel = mat.shape
    scale = 4
    hr = range(0, h * scale, scale)
    wr = range(0, w * scale, scale)
    axis = product2(hr, wr).astype(np.float32)
    return solve_pnp(mat.reshape(-1, 3), axis, intrinsics_mat.astype(np.float32))


def homo_transform(transform: np.ndarray):
    id_mat = np.identity(4)
    id_mat[:-1, :-1] = transform
    return id_mat


def homo_translation(translation: np.ndarray):
    id_mat = np.identity(4)
    id_mat.T[3, :-1] = translation.squeeze()
    return id_mat


def combine_rotation_translation(rotation: np.ndarray, translation: np.ndarray):
    rot_mat = cv.Rodrigues(rotation)[0]
    rot_homo = homo_transform(rot_mat)
    trans = homo_translation(translation)
    return np.matmul(rot_homo, trans)
