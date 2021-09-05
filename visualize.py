import numpy as np
import torch
from torch.utils import data
from pathlib import Path

from torch.utils.data import DataLoader

from data.sevenscenes import Scenes, Modes, SevenScenes, camera_focus
from torchvision import transforms

from data.worldpos import pixel_to_world_pos_tensor
from train import to_fucking_float_tensor
from validation.projection import scale_focus, intrinsics_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler


def main():
    from data.visutil import show_batch, show_stereo_batch
    from torchvision.utils import make_grid
    import torchvision.transforms as transforms
    from PIL.Image import Image
    seq: Scenes = 'chess'
    mode: Modes = 2
    num_workers = 1
    # num_workers = 6
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        # FIXME: Numpy complains __array__ takes 1 arg but 2 were given, fix.
        # Image.__array__,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    p = Path('..') / '..' / 'Datasets' / '7scenes'
    dset = SevenScenes(seq, p, True, transform,
                       mode=mode)
    print('Loaded 7Scenes sequence {:s}, length = {:d}'.format(seq,
                                                               len(dset)))

    data_loader = data.DataLoader(dset, batch_size=10, shuffle=True,
                                  num_workers=num_workers)

    batch_count = 0
    N = 2
    for batch in data_loader:
        print('Minibatch {:d}'.format(batch_count))
        if mode < 2:
            show_batch(make_grid(batch[0], nrow=1, padding=25, normalize=True))
        elif mode == 2:
            lb = make_grid(batch[0][0], nrow=1, padding=25, normalize=True)
            rb = make_grid(batch[0][1], nrow=1, padding=25, normalize=True)
            show_stereo_batch(lb, rb)

        batch_count += 1
        if batch_count >= N:
            break


def analyse_minmax():
    from config import scene, mode, num_workers, path, Globals
    from tqdm import tqdm
    # scene: Scenes = 'chess'
    # mode: Modes = 2
    # num_workers = 4
    # path = Path(__file__).absolute().parent.parent.parent / 'Datasets' / '7scenes'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(256),
        # Image.__array__,
        transforms.ToTensor(),
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

    def target_trans(tensor: torch.Tensor, pos: np.ndarray):
        return pixel_to_world_pos_tensor(tensor, pos, focus=focus, center=center)

    dataset = SevenScenes(
        scene, path, True, transform=transform, target_transform=target_trans, mode=mode
    )
    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=num_workers
    )
    scaler = RobustScaler()
    std_scaler = StandardScaler()
    batch = next(iter(loader))
    [rgb_image, _depth_image], world_point = batch
    world_point = (to_fucking_float_tensor(world_point))
    out_pts = scaler.fit_transform(world_point.view(-1, 3))
    std_scaler.fit_transform(out_pts)

    return scaler, std_scaler


if __name__ == '__main__':
    main()
