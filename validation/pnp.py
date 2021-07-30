import cv2 as cv
import numpy as np
from numpy.linalg import inv as inverse_matrix
import transforms3d.quaternions
import math
from numpy.typing import ArrayLike

from data.worldpos import product2, to_homo, to_cartesian
from .projection import default_intrinsics_matrix
from transforms3d import quaternions as tsq
from scipy.spatial.transform.rotation import Rotation


def solve_pnp(world_pos: np.ndarray, pixel_pos: np.ndarray, intrinsics_mat: np.ndarray):
    _retval, rvecs, tvecs, _inliers = cv.solvePnPRansac(world_pos, pixel_pos, intrinsics_mat, np.zeros(5))
    return rvecs, tvecs


def camera_pos_from_output(mat: np.ndarray, depth_map: ArrayLike, intrinsics_mat: np.ndarray = default_intrinsics_matrix) -> (np.ndarray, np.ndarray):
    """
    Compute estimated extrinsics matrix. i.e. inv(camera_to_world)
    Args:
        mat: The model output. 0.25x sampled pixel_pos => world_pos mapping.
        depth_map: The original depth image.
        intrinsics_mat: Literally, camera intrinsics matrix.

    Returns: (rotation vector, translation vector)
    """

    h, w, *_channel = mat.shape
    scale = 4

    # The x axis of sampled coords.
    hr = range(0, h * scale, scale)
    # The y axis of sampled coords.
    wr = range(0, w * scale, scale)
    # To get Z_c, we need to sample depth image.
    sampled_depth = depth_map[np.ix_(hr, wr)]
    # Combine Grid x and y axis.
    # pts is the sampled points (in pixel coords). (u, v)
    pts = np.array(np.meshgrid(hr, wr)).T.reshape(len(hr), len(wr), 2)
    # (u, v, 1)
    pos = to_homo(pts)
    # Z_c * (u, v, 1)
    homo_pixel = pos * np.expand_dims(sampled_depth, 2)

    # camera_pos = pixel_pos(T) * inv(intrinsics).T
    camera_pos = np.matmul(homo_pixel, inverse_matrix(intrinsics_mat).T)
    camera_pos[..., -1][camera_pos[..., -1] == 0] = 1

    # sampled points (in camera pos)
    axis = to_cartesian(camera_pos)
    # SolvePnPRansac need vector of points, just reshape them.
    return solve_pnp(mat.reshape(-1, 3), axis.reshape(-1, 2), intrinsics_mat.astype(np.float32))


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
    return np.matmul(trans, rot_homo)


def decompose_transform(matrix: np.ndarray):
    assert matrix.shape == (4, 4)
    assert matrix[-1, -1] == 1.0
    rot = matrix[:3, :3]
    translate = matrix[:3, 3]
    qut = transforms3d.quaternions.mat2quat(rot)
    return qut, translate


def qut_error(pose_q: ArrayLike, predicted_q: ArrayLike):
    # Compute Individual Sample Error
    q1 = pose_q / np.linalg.norm(pose_q)
    q2 = predicted_q / np.linalg.norm(predicted_q)
    d = abs(np.sum(np.multiply(q1, q2)))
    theta = 2 * np.arccos(d) * 180 / math.pi
    return theta


def trans_error(pose_t: ArrayLike, predicted_t: ArrayLike):
    return np.linalg.norm(np.asarray(pose_t) - np.asarray(predicted_t))
