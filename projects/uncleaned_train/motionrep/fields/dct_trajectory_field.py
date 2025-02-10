import torch
import torch.nn as nn

from motionrep.utils.dct import dct, idct, dct3d, idct_3d


class DCTTrajctoryField(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def query_points_at_time(self, x, t):
        pass
