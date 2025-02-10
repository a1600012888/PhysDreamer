import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
from typing import Optional, Sequence, Tuple, List
from motionrep.field_components.encoding import TriplanesEncoding
from motionrep.field_components.mlp import MLP
from motionrep.data.scene_box import SceneBox


class TriplaneFields(nn.Module):
    """Temporal Kplanes SE(3) fields.

    Args:
        aabb: axis-aligned bounding box.
            aabb[0] is the minimum (x,y,z) point.
            aabb[1] is the maximum (x,y,z) point.
        resolutions: resolutions of the kplanes. in an order of [x, y, z]

    """

    def __init__(
        self,
        aabb: Float[Tensor, "2 3"],
        resolutions: Sequence[int],
        feat_dim: int = 64,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce="sum",  #: Literal["sum", "product", "cat"] = "sum",
        num_decoder_layers=2,
        decoder_hidden_size=64,
        output_dim: int = 96,
        zero_init: bool = False,
    ):
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.output_dim = output_dim

        self.kplanes_encoding = TriplanesEncoding(
            resolutions, feat_dim, init_a, init_b, reduce
        )

        self.decoder = MLP(
            feat_dim,
            num_decoder_layers,
            layer_width=decoder_hidden_size,
            out_dim=self.output_dim,
            skip_connections=None,
            activation=nn.ReLU(),
            out_activation=None,
            zero_init=zero_init,
        )

    def forward(
        self, inp: Float[Tensor, "*bs 3"]
    ) -> Tuple[Float[Tensor, "*bs 3 3"], Float[Tensor, "*bs 3"]]:
        # shift to [-1, 1]
        inpx = SceneBox.get_normalized_positions(inp, self.aabb) * 2.0 - 1.0

        output = self.kplanes_encoding(inpx)

        output = self.decoder(output)

        # split_size = output.shape[-1] // 3
        # output = torch.stack(torch.split(output, split_size, dim=-1), dim=-1)

        return output

    def compute_smoothess_loss(
        self,
    ):
        smothness_loss = self.kplanes_encoding.compute_plane_tv()

        return smothness_loss


def compute_entropy(p):
    return -torch.sum(p * torch.log(p + 1e-5), dim=1).mean()  # Adding a small constant to prevent log(0)


class TriplaneFieldsWithEntropy(nn.Module):
    """Temporal Kplanes SE(3) fields.

    Args:
        aabb: axis-aligned bounding box.
            aabb[0] is the minimum (x,y,z) point.
            aabb[1] is the maximum (x,y,z) point.
        resolutions: resolutions of the kplanes. in an order of [x, y, z]

    """

    def __init__(
        self,
        aabb: Float[Tensor, "2 3"],
        resolutions: Sequence[int],
        feat_dim: int = 64,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce="sum",  #: Literal["sum", "product", "cat"] = "sum",
        num_decoder_layers=2,
        decoder_hidden_size=64,
        output_dim: int = 96,
        zero_init: bool = False,
        num_cls: int = 3,
    ):
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.output_dim = output_dim
        self.num_cls = num_cls

        self.kplanes_encoding = TriplanesEncoding(
            resolutions, feat_dim, init_a, init_b, reduce
        )

        self.decoder = MLP(
            feat_dim,
            num_decoder_layers,
            layer_width=decoder_hidden_size,
            out_dim=self.num_cls,
            skip_connections=None,
            activation=nn.ReLU(),
            out_activation=None,
            zero_init=zero_init,
        )

        self.cls_embedding = torch.nn.Embedding(num_cls, output_dim)

    def forward(
        self, inp: Float[Tensor, "*bs 3"]
    ) -> Tuple[Float[Tensor, "*bs 3 3"], Float[Tensor, "1"]]:
        # shift to [-1, 1]
        inpx = SceneBox.get_normalized_positions(inp, self.aabb) * 2.0 - 1.0

        output = self.kplanes_encoding(inpx)

        output = self.decoder(output)

        prob = F.softmax(output, dim=-1)

        entropy = compute_entropy(prob)

        cls_index = torch.tensor([0, 1, 2]).to(inp.device)
        cls_emb = self.cls_embedding(cls_index)

        output = torch.matmul(prob, cls_emb)

        
        return output, entropy

    def compute_smoothess_loss(
        self,
    ):
        smothness_loss = self.kplanes_encoding.compute_plane_tv()

        return smothness_loss
