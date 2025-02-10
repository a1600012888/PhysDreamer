import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
from typing import Literal, Optional, Sequence, Tuple
from motionrep.field_components.encoding import (
    TemporalKplanesEncoding,
    TriplanesEncoding,
)
from motionrep.field_components.mlp import MLP
from motionrep.operators.rotation import rotation_6d_to_matrix, quaternion_to_matrix
from motionrep.data.scene_box import SceneBox


class TemporalKplanesSE3fields(nn.Module):
    """Temporal Kplanes SE(3) fields.

    Args:
        aabb: axis-aligned bounding box.
            aabb[0] is the minimum (x,y,z) point.
            aabb[1] is the maximum (x,y,z) point.
        resolutions: resolutions of the kplanes. in an order of [x, y, z ,t].

    """

    def __init__(
        self,
        aabb: Float[Tensor, "2 3"],
        resolutions: Sequence[int],
        feat_dim: int = 64,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce: Literal["sum", "product", "cat"] = "sum",
        num_decoder_layers=2,
        decoder_hidden_size=64,
        rotation_type: Literal["quaternion", "6d"] = "6d",
        add_spatial_triplane: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()

        self.register_buffer("aabb", aabb)
        output_dim_dict = {"quaternion": 4 + 3, "6d": 6 + 3}
        self.output_dim = output_dim_dict[rotation_type]
        self.rotation_type = rotation_type

        self.temporal_kplanes_encoding = TemporalKplanesEncoding(
            resolutions, feat_dim, init_a, init_b, reduce
        )

        self.add_spatial_triplane = add_spatial_triplane
        if add_spatial_triplane:
            self.spatial_kplanes_encoding = TriplanesEncoding(
                resolutions[:-1], feat_dim, init_a, init_b, reduce
            )
            feat_dim = feat_dim * 2

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
        self,
        inp: Float[Tensor, "*bs 4"],
        compute_smoothess_loss: bool = False,
    ) -> Tuple[Float[Tensor, "*bs 3 3"], Float[Tensor, "*bs 3"]]:
        if compute_smoothess_loss:
            smothness_loss, temporal_smoothness_loss = self.compute_smoothess_loss()
            return smothness_loss + temporal_smoothness_loss
        inpx, inpt = inp[:, :3], inp[:, 3:]

        # shift to [-1, 1]
        inpx = SceneBox.get_normalized_positions(inpx, self.aabb) * 2.0 - 1.0

        inpt = inpt * 2.0 - 1.0

        inp = torch.cat([inpx, inpt], dim=-1)
        output = self.temporal_kplanes_encoding(inp)

        if self.add_spatial_triplane:
            spatial_output = self.spatial_kplanes_encoding(inpx)
            output = torch.cat([output, spatial_output], dim=-1)

        output = self.decoder(output)

        if self.rotation_type == "6d":
            rotation_6d, translation = output[:, :6], output[:, 6:]
            R_mat = rotation_6d_to_matrix(rotation_6d)

        elif self.rotation_type == "quaternion":
            quat, translation = output[:, :4], output[:, 4:]

            # tanh and normalize
            quat = torch.tanh(quat)

            R_mat = quaternion_to_matrix(quat)

            # --------------- remove below --------------- #
            # add normalization
            # r = quat
            # norm = torch.sqrt(
            #     r[:, 0] * r[:, 0]
            #     + r[:, 1] * r[:, 1]
            #     + r[:, 2] * r[:, 2]
            #     + r[:, 3] * r[:, 3]
            # )
            # q = r / norm[:, None]
            # R_mat = q
            # --------------- remove above --------------- #

        return R_mat, translation

    def compute_smoothess_loss(
        self,
    ):
        smothness_loss = self.temporal_kplanes_encoding.compute_plane_tv()
        temporal_smoothness_loss = (
            self.temporal_kplanes_encoding.compute_temporal_smoothness()
        )

        if self.add_spatial_triplane:
            smothness_loss += self.spatial_kplanes_encoding.compute_plane_tv()

        return smothness_loss, temporal_smoothness_loss

    def compute_loss(
        self,
        inp: Float[Tensor, "*bs 4"],
        trajectory: Float[Tensor, "*bs 3"],
        loss_func,
    ):
        inpx, inpt = inp[:, :3], inp[:, 3:]

        R, t = self(inp)

        rec_traj = torch.bmm(R, inpx.unsqueeze(-1)).squeeze(-1) + t

        rec_loss = loss_func(rec_traj, trajectory)

        return rec_loss
