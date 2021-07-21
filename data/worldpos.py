import torch
from PIL.Image import Image
import numpy as np
from validation.projection import Point2d, pixel_to_camera_matrix
from typing import Iterable


def assoc_z_axis(ndi: np.ndarray, homo: bool=True):
    @np.vectorize
    def iter():
        ys = []
        for i, row in enumerate(ndi):
            xs = []
            for j, depth in enumerate(row):
                coord = [i, j, depth]
                if homo:
                    coord.append(1)
                xs.append(coord)
            ys.append(xs)
        return ys
    return np.array(iter())


def assoc_z_axis_cartesian(ndi: np.ndarray):
    r, c = ndi.shape
    pts = np.array(np.meshgrid(range(r), range(c))).T.reshape(r, c, 2)
    return np.concatenate((pts, np.expand_dims(ndi, axis=2)), axis=2)


def product2(it1: Iterable, it2: Iterable) -> np.ndarray:
    return np.array(np.meshgrid(it1, it2)).T.reshape(-1, 2)


def get_world_pos(dimage: Image, pos: np.ndarray, scale_ratio: int = 4):
    w, h = dimage.size
    index = np.ix_(range(0, h, scale_ratio), range(0, w, scale_ratio))
    # color_array = np.array(image)[index]
    assert dimage.mode == 'I'
    depth_array = np.array(dimage)[index]
    homo_coords = assoc_z_axis(depth_array, homo=True)
    return np.dot(homo_coords, pos)


def to_pixel_pos(dimage: Image) -> np.ndarray:
    assert dimage.mode == 'I'
    depth_array = np.array(dimage)
    return assoc_z_axis_cartesian(depth_array)


def to_camera_pos(pixel_pos: np.ndarray, focus: Point2d, center: Point2d) -> np.ndarray:
    pixel_to_camera = pixel_to_camera_matrix(focus, center)
    return np.matmul(pixel_pos, pixel_to_camera)


def down_sampling(img: np.ndarray, scale_ratio: int = 4):
    h, w, _channel = img.shape
    index = np.ix_(range(0, h, scale_ratio), range(0, w, scale_ratio))
    return img[index]


def to_homo(array: np.ndarray) -> np.ndarray:
    h, w, d = array.shape
    return np.insert(array, d, 1, axis=2)


def to_cartesian(coord: np.ndarray) -> np.ndarray:
    return np.delete(coord / np.expand_dims(coord[..., 3], axis=2), 3, 2)


def camera_to_world_pos(img: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    homo_img = to_homo(img)
    return np.matmul(homo_img, camera_to_world)


def get_world_pos_tensor(tensor: torch.Tensor, pos: np.ndarray, scale_ratio: int = 4):
    _dimension, h, w = tensor.shape
    index = np.ix_(range(0, h, scale_ratio), range(0, w, scale_ratio))
    depth_array = tensor.numpy().squeeze()[index]
    homo_coords = assoc_z_axis(depth_array, homo=True)
    transformed = np.dot(homo_coords, pos)
    return torch.from_numpy(transformed)


def pixel_to_world_pos_tensor(
        tensor: torch.Tensor, camera_to_world: np.ndarray, focus: Point2d, center: Point2d, scale_ratio: int = 4
) -> torch.Tensor:
    # w * h * 1 matrix -> w * h * 3
    # [[[1]]] -> [[[0, 0, 1]]] with its coordinates
    pixel_pos = assoc_z_axis_cartesian(tensor.numpy().squeeze())
    camera_pos = to_camera_pos(pixel_pos, focus, center)
    camera_pos = down_sampling(camera_pos, scale_ratio)
    world_pos = camera_to_world_pos(camera_pos, camera_to_world)
    cartesian_world_pos = to_cartesian(world_pos)
    return torch.from_numpy(cartesian_world_pos)


def reverse_diem(tensor: torch.Tensor) -> torch.Tensor:
    batch, *shape = tensor.shape
    return tensor.reshape(batch, *shape[::-1])
