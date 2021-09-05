"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torchvision.datasets.folder import default_loader
import torch
import numpy as np


def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None

    return img


def pil_to_tensor_without_fucking_permuting(pic):
    """Convert a ``PIL Image`` to a tensor of the same type.
    This function does not support torchscript.

    Args:
        pic (PIL Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """

    # handle PIL Image
    img = torch.as_tensor(np.array(pic))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    return img


def rgb_to_tensor_permuting(tens: torch.Tensor) -> torch.Tensor:
    dtype = torch.get_default_dtype()
    tens = tens.permute((0, 3, 1, 2)) # type: torch.Tensor
    return tens.contiguous().to(dtype=dtype).div(255)
