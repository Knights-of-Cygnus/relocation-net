from pathlib import Path
from data.sevenscenes import Scenes, Modes
from numpy.typing import ArrayLike


scene: Scenes = 'chess'
mode: Modes = 2
num_workers = 4
path = Path(__file__).absolute().parent.parent.parent / 'Datasets' / '7scenes'


class Globals:
    intrinsics_mat: ArrayLike = None
    test_dataloader = None
    focus = None
    center = None
