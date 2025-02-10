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


class MulTemporalKplanesSE3fields(nn.Module):
    """Multiple Temporal Kplanes SE(3) fields.

    Args:
        aabb: axis-aligned bounding box.
            aabb[0] is the minimum (x,y,z) point.
            aabb[1] is the maximum (x,y,z) point.
        resolutions: resolutions of the kplanes. in an order of [x, y, z ,t].

    """

    def __init__(
        self,
        aabb: Float[Tensor, "2 3"],
        resolutions_list: Sequence[int],
        feat_dim: int = 64,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce: Literal["sum", "product", "cat"] = "sum",
        num_decoder_layers=2,
        decoder_hidden_size=64,
        rotation_type: Literal["quaternion", "6d"] = "6d",
        add_spatial_triplane: bool = True,
    ):
        super().__init__()

        self.register_buffer("aabb", aabb)
        output_dim_dict = {"quaternion": 4 + 3, "6d": 6 + 3}
        self.output_dim = output_dim_dict[rotation_type]
        self.rotation_type = rotation_type

        self.temporal_kplanes_encoding_list = nn.ModuleList(
            [
                TemporalKplanesEncoding(resolutions, feat_dim, init_a, init_b, reduce)
                for resolutions in resolutions_list
            ]
        )

        self.add_spatial_triplane = add_spatial_triplane
        if add_spatial_triplane:
            self.spatial_kplanes_encoding_list = nn.ModuleList(
                [
                    TriplanesEncoding(
                        resolutions[:-1], feat_dim, init_a, init_b, reduce
                    )
                    for resolutions in resolutions_list
                ]
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
        )

    def forward(
        self, inp: Float[Tensor, "*bs 4"], dataset_indx: Int[Tensor, "1"]
    ) -> Tuple[Float[Tensor, "*bs 3 3"], Float[Tensor, "*bs 3"]]:
        inpx, inpt = inp[:, :3], inp[:, 3:]

        # shift to [-1, 1]
        inpx = SceneBox.get_normalized_positions(inpx, self.aabb) * 2.0 - 1.0

        inpt = inpt * 2.0 - 1.0

        inp = torch.cat([inpx, inpt], dim=-1)

        # for loop in batch dimension

        output = self.temporal_kplanes_encoding_list[dataset_indx](inp)

        if self.add_spatial_triplane:
            spatial_output = self.spatial_kplanes_encoding_list[dataset_indx](inp)
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

        return R_mat, translation

    def compute_smoothess_loss(
        self,
    ):
        temporal_smoothness_loss = 0.0
        for temporal_kplanes_encoding in self.temporal_kplanes_encoding_list:
            temporal_smoothness_loss += (
                temporal_kplanes_encoding.compute_temporal_smoothness()
            )

        smothness_loss = 0.0
        for temporal_kplanes_encoding in self.temporal_kplanes_encoding_list:
            smothness_loss += temporal_kplanes_encoding.compute_plane_tv()

        if self.add_spatial_triplane:
            for spatial_kplanes_encoding in self.spatial_kplanes_encoding_list:
                smothness_loss += spatial_kplanes_encoding.compute_plane_tv()

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
