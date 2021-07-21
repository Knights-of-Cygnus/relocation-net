import torch
import torch.nn as nn
from .segnet import SegNet

class ShitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.segnet = SegNet(
            input_channels=3,
            output_channels=3
        )
        
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.segnet(data)
