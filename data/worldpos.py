import torch
from PIL.Image import Image
import numpy as np
from validation.projection import Point2d, pixel_to_camera_matrix
from typing import Iterable
from deprecation import deprecated
from torchvision import transforms

invalid_depth = 65535


def transform_with_inverse(ts):
    norms = []
    for trans in reversed(ts):
        std = trans.std
        mean = trans.mean
        norms.append(transforms.Normalize(
            std=[1 / x for x in std],
            mean=[0 for x in mean]
        ))
        norms.append(transforms.Normalize(
            std=[1 for x in std],
            mean=[-x for x in mean]
        ))

    return transforms.Compose(ts), transforms.Compose(norms)


world_point_transform, inverse_world_point_transform = transform_with_inverse([
    transforms.Normalize(
        std=[1.04586485, 0.92645615, 1.72081834],
        mean=[-0.03427729, -0.19971056,  1.48099369]
    )
])


@deprecated
def assoc_z_axis(ndi: np.ndarray, homo: bool = True):
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


@deprecated
def assoc_z_axis_cartesian(ndi: np.ndarray) -> np.ndarray:
    r, c = ndi.shape
    pts = np.array(np.meshgrid(range(r), range(c))).T.reshape(r, c, 2)
    homo_pixel = to_homo(pts)
    return ndi.reshape((r, c, 1)) * homo_pixel
    # return np.concatenate((pts, np.expand_dims(ndi, axis=2)), axis=2)


def product2(it1: Iterable, it2: Iterable) -> np.ndarray:
    return np.array(np.meshgrid(it1, it2)).T.reshape(-1, 2)


def tabulate2(row: int, col: int) -> np.ndarray:
    return product2(range(row), range(col)).reshape((row, col, 2))


@deprecated
def get_world_pos(dimage: Image, pos: np.ndarray, scale_ratio: int = 4):
    w, h = dimage.size
    index = np.ix_(range(0, h, scale_ratio), range(0, w, scale_ratio))
    # color_array = np.array(image)[index]
    assert dimage.mode == 'I'
    depth_array = np.array(dimage)[index]
    homo_coords = assoc_z_axis(depth_array, homo=True)
    return np.matmul(homo_coords, pos.T)


def to_pixel_pos(dimage: Image) -> np.ndarray:
    assert dimage.mode == 'I'
    depth_array = np.array(dimage)
    return assoc_z_axis_cartesian(depth_array)


def to_camera_pos(pixel_pos: np.ndarray, depths: np.ndarray, focus: Point2d, center: Point2d) -> np.ndarray:
    pixel_to_camera = pixel_to_camera_matrix(focus, center)
    # Truncate invalid depth
    depths = depths.copy() # type: np.ndarray
    depths[depths >= invalid_depth] = 0
    # Convert depths from mm to m
    depths = depths / 1000
    return np.matmul(pixel_pos * np.expand_dims(depths, 2), pixel_to_camera.T)


def down_sampling(img: np.ndarray, scale_ratio: int = 4):
    h, w, _channel = img.shape
    index = np.ix_(range(0, h, scale_ratio), range(0, w, scale_ratio))
    return img[index]


def to_homo(array: np.ndarray) -> np.ndarray:
    h, w, d = array.shape
    return np.insert(array, d, 1, axis=2)


def to_cartesian(coord: np.ndarray) -> np.ndarray:
    last_axis = len(coord.shape) - 1
    last_pos = coord.shape[last_axis] - 1
    return np.delete(coord / np.expand_dims(coord[..., last_pos], axis=last_axis), last_pos, last_axis)


def camera_to_world_pos(img: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    homo_img = to_homo(img)
    return np.matmul(homo_img, camera_to_world.T)


def get_world_pos_tensor(tensor: torch.Tensor, pos: np.ndarray, scale_ratio: int = 4):
    _dimension, h, w = tensor.shape
    index = np.ix_(range(0, h, scale_ratio), range(0, w, scale_ratio))
    depth_array = tensor.numpy().squeeze()[index]
    homo_coords = assoc_z_axis(depth_array, homo=True)
    transformed = np.dot(homo_coords, pos)
    return torch.from_numpy(transformed)


def pixel_to_world_pos_tensor_pixelwise(
        tensor: torch.Tensor, camera_to_world: np.ndarray, focus: Point2d, center: Point2d, scale_ratio: int = 4
) -> torch.Tensor:
    # w * h * 1 matrix -> w * h * 3
    # [[[1]]] -> [[[0, 0, 1]]] with its coordinates
    # Pixel_pos = Z_c * (u, v, 1)
    # pixel_pos = assoc_z_axis_cartesian(tensor.numpy().squeeze())
    # Clamp invalid numbers.
    # if Z_c = 0 then last coord must be 0.
    # In such case, just override it to 1.
    # pixel_pos[..., -1][pixel_pos[..., -1] == 0] = 1

    # camera_pos = pixel_pos.T * inv(intrinsics_matrix).T
    # camera_pos = to_camera_pos(pixel_pos, focus, center)
    # camera_pos = depth_to_camera_pixelwise(tensor.numpy().squeeze(), center, focus)

    # down sampling by scale ratio
    # camera_pos = down_sampling(camera_pos, scale_ratio)

    # world_pos = (camera_pos, 1) * camera_to_world.T
    # world_pos = camera_to_world_pos(camera_pos, camera_to_world)

    # Transform back to cartesian coords.
    # cartesian_world_pos = to_cartesian(world_pos)
    cartesian_world_pos = depth_to_world_pixelwise(
        tensor.numpy(), center, focus, camera_to_world
    )
    cartesian_world_pos = down_sampling(cartesian_world_pos, scale_ratio)
    return torch.from_numpy(cartesian_world_pos)


def pixel_to_world_pos_tensor(
        tensor: torch.Tensor, camera_to_world: np.ndarray, focus: Point2d, center: Point2d, scale_ratio: int = 4
) -> torch.Tensor:
    # w * h * 1 matrix -> w * h * 3
    # [[[1]]] -> [[[0, 0, 1]]] with its coordinates
    # Pixel_pos = (u, v, 1)
    depths = tensor.numpy().squeeze()  # type: np.ndarray
    row, col = depths.shape
    pixel_pos = to_homo(tabulate2(row, col))
    # Clamp invalid numbers.
    # if Z_c = 0 then last coord must be 0.
    # In such case, just override it to 1.
    # pixel_pos[..., -1][pixel_pos[..., -1] == 0] = 1

    # camera_pos = pixel_pos.T * inv(intrinsics_matrix).T
    camera_pos = to_camera_pos(pixel_pos, depths, focus, center)
    # camera_pos = depth_to_camera_pixelwise(tensor.numpy().squeeze(), center, focus)

    # down sampling by scale ratio
    camera_pos = down_sampling(camera_pos, scale_ratio)

    # world_pos = (camera_pos, 1) * camera_to_world.T
    world_pos = camera_to_world_pos(camera_pos, camera_to_world)

    # Transform back to cartesian coords.
    cartesian_world_pos = to_cartesian(world_pos)
    # cartesian_world_pos = depth_to_world_pixelwise(
    #     tensor.numpy(), center, focus, camera_to_world
    # )
    return torch.from_numpy(cartesian_world_pos)


def depth_to_world_pixelwise(dimage: np.ndarray, center: Point2d, focus: Point2d,
                             camera_to_world: np.ndarray) -> np.ndarray:
    arrays = []
    r, c = dimage.shape
    for u, row in enumerate(dimage):
        for v, z in enumerate(row):
            x, y, z = one_pixel_to_camera(u, v, z, center, focus)
            camera = np.array([[x, y, z, 1]]).T
            world = np.matmul(camera_to_world, camera).T.squeeze()
            x, y, z, w = world
            arrays.append(np.array([x / w, y / w, z / w]))

    return np.array(arrays).reshape((r, c, 3))


def one_pixel_to_camera(u, v, z, center: Point2d, focus: Point2d) -> list:
    cx, cy = center
    fx, fy = focus
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return [x, y, z]
