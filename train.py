import torch
from torch import nn
import time
import os
from torch.utils.data import DataLoader
from data.sevenscenes import SevenScenes, Scenes, Modes, camera_focus
from models.shitnet import ShitNet
from data.utils import pil_to_tensor_without_fucking_permuting, rgb_to_tensor_permuting
from data.worldpos import pixel_to_world_pos_tensor, world_point_transform, inverse_world_point_transform, down_sampling
from validation.projection import scale_focus, intrinsics_matrix
from pathlib import Path
from torchvision.transforms import transforms
import numpy as np
from numpy.linalg import inv as inverse_matrix
from validation.pnp import decompose_transform, camera_pos_from_output, trans_error, qut_error, combine_rotation_translation
from tqdm.notebook import tqdm

NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = 64

NUM_EPOCHS = 6000

LEARNING_RATE = 1e-4
MOMENTUM = 0.9
BATCH_SIZE = 16
SAVE_DIR = Path(__file__).parent / 'checkpoints'
LAST_EPOCH = 0

model = None


def SoftXEnt(input: torch.Tensor, target: torch.Tensor):
    return -(target * input).sum() / input.shape[0]


def to_fucking_float_tensor(img: torch.Tensor):
    dtype = torch.get_default_dtype()
    return img.to(dtype)


@torch.no_grad()
def test_validation(model: nn.Module, device: torch.device):
    from config import Globals
    total_rot_error = 0.0
    total_trans_error = 0.0
    count = 0
    for batch in tqdm(Globals.test_dataloader):
        [rgb_image, depth_image], world_point = batch
        rgb_image = rgb_to_tensor_permuting(rgb_image)
        predicted, _softmaxed = model(rgb_image.to(device))  # type: (torch.Tensor, torch.Tensor)
        predicted = inverse_world_point_transform(predicted) # type: torch.Tensor
        predicted = predicted.permute((0, 2, 3, 1))
        assert predicted.shape[0] == world_point.shape[0]
        for p, w in zip(predicted.cpu().numpy(), world_point.numpy()):
            rvec, tvec = camera_pos_from_output(p, depth_image, Globals.intrinsics_mat)
            pos = combine_rotation_translation(np.asarray(rvec), np.asarray(tvec))
            pos = inverse_matrix(pos)
            est_rot, est_trans = decompose_transform(pos)
            tar_rot, tar_trans = decompose_transform(w)
            error_r = qut_error(est_rot, tar_rot)
            error_t = trans_error(est_trans, tar_trans)
            total_rot_error += error_r
            total_trans_error += error_t
            count += 1

    return total_rot_error / count, total_trans_error / count


@torch.no_grad()
def test_loss(model: nn.Module, device: torch.device) -> float:
    from config import Globals
    total_loss = 0
    loss_fn = torch.nn.L1Loss().to(device)
    for batch in tqdm(Globals.test_dataloader):
        [rgb_image, depth_image], points = batch
        points = world_point_transform(points.permute(0, 3, 1, 2))
        points = points.to(device)
        # points.apply_(world_point_transform)
        rgb_image = rgb_to_tensor_permuting(rgb_image)
        predicted, softmaxed = model(rgb_image.to(device))
        total_loss += loss_fn(points, predicted)
    return total_loss


def projection_loss(est: torch.Tensor, act: torch.Tensor) -> float:
    pass


def train(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    is_better = True
    prev_loss = float('inf')

    prev_rot_error = float('inf')
    prev_trans_error = float('inf')

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    # criterion = torch.nn.MSELoss().to(device)
    criterion = torch.nn.L1Loss().to(device)

    model.train()
    model.to(device)

    for epoch in range(LAST_EPOCH, LAST_EPOCH + NUM_EPOCHS):
        loss_f = 0
        t_start = time.time()

        for batch_id, batch in tqdm(enumerate(train_dataloader), total=4000 / train_dataloader.batch_size):
            [rgb_image, _depth_image], world_point = batch
            rgb_image = rgb_to_tensor_permuting(rgb_image)
            world_point = to_fucking_float_tensor(world_point).permute(0, 3, 1, 2)
            world_point = world_point_transform(world_point)
            input_tensor = torch.autograd.Variable(rgb_image)
            target_tensor = torch.autograd.Variable(world_point)

            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            predicted_tensor, softmaxed_tensor = model(input_tensor)

            optimizer.zero_grad()
            loss = criterion(predicted_tensor, target_tensor)
            loss.backward()
            optimizer.step()

            loss_f += loss.float()
            prediction_f = softmaxed_tensor.float()

        is_better = loss_f < prev_loss
        loss_test = test_loss(model, device)
        print(f"Epoch #{epoch + 1}: Test Loss: {loss_test}")
        # error_r, error_t = test_validation(model, device)
        #
        delta = time.time() - t_start
        # if error_r < prev_rot_error or error_t < prev_trans_error:
        #     print(f'Epoch #{epoch} rot: {prev_rot_error} -> {error_r}, trnas: {prev_trans_error} -> {error_t}')
        #     prev_rot_error = error_r
        #     prev_trans_error = error_t
        #     torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'model_valid_best_{epoch}.pth'))

        if is_better:
            prev_loss = loss_f
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_best_epo{epoch}.pth"))

        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch + 1, loss_f, delta))


def train_shitnet():
    global model
    if model is None:
        model = ShitNet()
    from config import scene, mode, num_workers, path, Globals
    # scene: Scenes = 'chess'
    # mode: Modes = 2
    # num_workers = 4
    # path = Path(__file__).absolute().parent.parent.parent / 'Datasets' / '7scenes'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(256),
        # Image.__array__,
        # transforms.ToTensor(),
        pil_to_tensor_without_fucking_permuting,
        # to_fucking_float_tensor,
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
    ])

    w, h = 640, 480
    focus = scale_focus(camera_focus, (256 / h, 256 / w))
    center = (128, 128)
    Globals.intrinsics_mat = intrinsics_matrix(focus, center)
    Globals.center = center
    Globals.focus = focus

    def target_trans(tensor: torch.Tensor, pos: np.ndarray):
        return pixel_to_world_pos_tensor(tensor, pos, focus=focus, center=center)

    dataset = SevenScenes(
        scene, path, True, transform=transform, target_transform=target_trans, mode=mode
    )
    test_dataset = SevenScenes(
        scene, path, train=False, transform=transform, mode=mode
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers
    )
    Globals.test_dataloader = test_dataloader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers
    )

    train(model, loader, torch.device('cuda'))


def train_with_open3d():
    import open3d as o3d
    from config import scene, mode, num_workers, path, Globals
    global model
    if model is None:
        model = ShitNet()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(256),
        # Image.__array__,
        pil_to_tensor_without_fucking_permuting,
        # to_fucking_float_tensor,
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
    ])

    w, h = 640, 480
    focus = scale_focus(camera_focus, (256 / h, 256 / w))
    center = (128, 128)
    intr = o3d.camera.PinholeCameraIntrinsic(256, 256, *focus, *center)

    def target_trans(rgb_tensor: torch.Tensor, depth_tensor: torch.Tensor, pos: np.ndarray):
        rgb = o3d.geometry.Image(rgb_tensor.numpy())
        depth = o3d.geometry.Image(depth_tensor.numpy().astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intr, pos, False
        )
        points = np.asarray(pcd.points)
        full_pos = np.nan_to_num(points).reshape(rgb_tensor.shape)
        return down_sampling(full_pos, 4)

    dataset = SevenScenes(
        scene, path, True, transform=transform, mode=mode, target_transform=target_trans
    )
    test_dataset = SevenScenes(
        scene, path, train=False, transform=transform, mode=mode, target_transform=target_trans
    )
    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers
    )
    Globals.test_dataloader = test_dataloader

    train(model, loader, torch.device('cuda'))


def load_model(path: str):
    global model
    if model is None:
        model = ShitNet()
    state = torch.load(path)
    model.load_state_dict(state)


if __name__ == '__main__':
    train_shitnet()
