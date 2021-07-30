import torch
from torch import nn
import time
import os
from torch.utils.data import DataLoader
from data.sevenscenes import SevenScenes, Scenes, Modes, camera_focus
from models.shitnet import ShitNet
from data.worldpos import pixel_to_world_pos_tensor, reverse_diem
from validation.projection import scale_focus
from pathlib import Path
from torchvision.transforms import transforms
import numpy as np
from PIL.Image import Image

NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = 64

NUM_EPOCHS = 6000

LEARNING_RATE = 1e-3
MOMENTUM = 0.9
BATCH_SIZE = 16
SAVE_DIR = Path(__file__).parent / 'checkpoints'


def SoftXEnt(input: torch.Tensor, target: torch.Tensor):
    return -(target * input).sum() / input.shape[0]


def to_fucking_float_tensor(img: torch.Tensor):
    dtype = torch.get_default_dtype()
    return img.to(dtype)


def train(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    is_better = True
    prev_loss = float('inf')

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion = torch.nn.MSELoss().to(device)

    model.train()
    model.to(device)

    for epoch in range(NUM_EPOCHS):
        loss_f = 0
        t_start = time.time()

        for batch_id, batch in enumerate(train_dataloader):
            [rgb_image, _depth_image], world_point = batch
            world_point = reverse_diem(to_fucking_float_tensor(world_point))
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

        delta = time.time() - t_start
        is_better = loss_f < prev_loss

        if is_better:
            prev_loss = loss_f
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_best_epo{epoch}.pth"))

        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch + 1, loss_f, delta))


def train_shitnet():
    model = ShitNet()
    from config import scene, mode, num_workers, path
    # scene: Scenes = 'chess'
    # mode: Modes = 2
    # num_workers = 4
    # path = Path(__file__).absolute().parent.parent.parent / 'Datasets' / '7scenes'
    transform = transforms.Compose([
        transforms.Resize(256),
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

    def target_trans(tensor: torch.Tensor, pos: np.ndarray):
        return pixel_to_world_pos_tensor(tensor, pos, focus=focus, center=(320, 240))

    dataset = SevenScenes(
        scene, path, True, transform=transform, target_transform=target_trans, mode=mode
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers
    )

    train(model, loader, torch.device('cuda'))


if __name__ == '__main__':
    train_shitnet()
