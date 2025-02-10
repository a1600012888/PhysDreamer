import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
from typing import Optional, Sequence, Tuple, List
from motionrep.field_components.encoding import (
    TriplanesEncoding,
    PlaneEncoding,
    TemporalNeRFEncoding,
)
from motionrep.field_components.mlp import MLP
from motionrep.data.scene_box import SceneBox
from einops import rearrange, repeat


class TriplaneDispFields(nn.Module):
    """Kplanes Displacement fields.
        [x, t, t] => [dx, dy]
    Args:
        aabb: axis-aligned bounding box.
            aabb[0] is the minimum (x,y,t) point.
            aabb[1] is the maximum (x,y,t) point.
        resolutions: resolutions of the kplanes. in an order of [x, y, t]

    """

    def __init__(
        self,
        aabb: Float[Tensor, "2 3"],
        resolutions: Sequence[int],
        feat_dim: int = 64,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce="cat",  #: Literal["sum", "product", "cat"] = "sum",
        num_decoder_layers=2,
        decoder_hidden_size=64,
        output_dim: int = 2,
        zero_init: bool = False,
    ):
        super().__init__()

        if aabb is None:
            aabb = (
                torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
                * 1.1
            )

        self.register_buffer("aabb", aabb)
        self.output_dim = output_dim

        self.canonical_encoding = PlaneEncoding(
            resolutions[:2], feat_dim, init_a, init_b
        )
        self.canonical_decoder = MLP(
            feat_dim,
            num_decoder_layers,
            layer_width=decoder_hidden_size,
            out_dim=3,
            skip_connections=None,
            activation=nn.ReLU(),
            out_activation=None,
        )

        self.kplanes_encoding = TriplanesEncoding(
            resolutions, feat_dim, init_a, init_b, reduce
        )

        if reduce == "cat":
            feat_dim = int(feat_dim * 3)

        self.decoder = MLP(
            feat_dim,
            int(num_decoder_layers * 3),
            layer_width=decoder_hidden_size,
            out_dim=self.output_dim,
            skip_connections=(2, 4),
            activation=nn.ReLU(),
            out_activation=None,
            zero_init=zero_init,
        )

    def forward(
        self, inp: Float[Tensor, "*bs 3"]
    ) -> Tuple[Float[Tensor, "*bs 2"], Float[Tensor, "*bs 3"]]:
        # shift to [-1, 1]
        inp_norm = SceneBox.get_normalized_positions(inp, self.aabb) * 2.0 - 1.0

        output = self.kplanes_encoding(inp_norm)

        # [*bs, 2]
        output = self.decoder(output)

        inpyx = inp_norm[..., :2].reshape(-1, 2)

        canonical_yx = inpyx + output

        ret_rgb_feat = self.canonical_encoding(canonical_yx)
        ret_rgb = self.canonical_decoder(ret_rgb_feat)

        return output, ret_rgb

    def compute_smoothess_loss(
        self,
    ):
        smothness_loss = self.kplanes_encoding.compute_plane_tv()

        smothness_canonical = self.canonical_encoding.compute_plane_tv()
        return smothness_loss + smothness_canonical

    def get_canonical(
        self, canonical_grid: Float[Tensor, "*bs 2"]
    ) -> Float[Tensor, "*bs 3"]:
        pad_can_grid = torch.cat(
            [canonical_grid, torch.zeros_like(canonical_grid[..., :1])], dim=-1
        )
        pad_can_norm = (
            SceneBox.get_normalized_positions(pad_can_grid, self.aabb) * 2.0 - 1.0
        )

        inp_can_grid = pad_can_norm[..., :2]

        ret_rgb_feat = self.canonical_encoding(inp_can_grid)
        ret_rgb = self.canonical_decoder(ret_rgb_feat)

        return ret_rgb

    def sample_canonical(
        self,
        inp: Float[Tensor, "bs hw 3"],
        canonical_frame: Float[Tensor, "1 H W 3"],
        canonical_grid_yx: Float[Tensor, "bs hw 2"],
    ) -> Float[Tensor, "bs h w 3"]:
        #
        inp_norm = SceneBox.get_normalized_positions(inp, self.aabb) * 2.0 - 1.0

        output = self.kplanes_encoding(inp_norm)

        # [-1, 2]
        output = self.decoder(output)

        inpyx = inp_norm[..., :2].reshape(-1, 2)

        canonical_yx = inpyx + output
        canonical_yx = canonical_yx * 1.1

        can_ymin, can_ymax = (
            canonical_grid_yx[..., 0].min(),
            canonical_grid_yx[..., 0].max(),
        )
        can_xmin, can_xmax = (
            canonical_grid_yx[..., 1].min(),
            canonical_grid_yx[..., 1].max(),
        )
        canonical_yx[..., 0] = (canonical_yx[..., 0] - can_ymin) / (
            can_ymax - can_ymin
        ) * 2.0 - 1.0
        canonical_yx[..., 1] = (canonical_yx[..., 1] - can_xmin) / (
            can_xmax - can_xmin
        ) * 2.0 - 1.0

        canonical_xy = torch.cat(
            [canonical_yx[..., 1:2], canonical_yx[..., 0:1]], dim=-1
        )
        # use grid sample to sample the canonical frame

        # [B, C, H, W]
        canonical_frame = canonical_frame.permute(0, 3, 1, 2).expand(
            inp.shape[0], -1, -1, -1
        )
        H, W = canonical_frame.shape[-2:]
        canonical_xy = canonical_xy.reshape(-1, H, W, 2)

        rec = F.grid_sample(canonical_frame, canonical_xy, align_corners=True)

        rec = rearrange(rec, "b c h w -> b h w c")

        return rec


class PlaneDynamicDispFields(nn.Module):
    """Plane Displacement fields.
        [x, t, t] => [dx, dy]
    Args:
        aabb: axis-aligned bounding box.
            aabb[0] is the minimum (x,y,t) point.
            aabb[1] is the maximum (x,y,t) point.
        resolutions: resolutions of the kplanes. in an order of [x, y, t]

    """

    def __init__(
        self,
        aabb: Float[Tensor, "2 3"],
        resolutions: Sequence[int],
        feat_dim: int = 64,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce="cat",  #: Literal["sum", "product", "cat"] = "sum",
        num_decoder_layers=2,
        decoder_hidden_size=64,
        output_dim: int = 2,
        zero_init: bool = False,
        num_temporal_freq: int = 20,
        freq_min: float = 0.0,
        freq_max: float = 20,
    ):
        super().__init__()

        if aabb is None:
            aabb = (
                torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
                * 1.1
            )

        self.register_buffer("aabb", aabb)
        self.output_dim = output_dim

        self.canonical_encoding = PlaneEncoding(
            resolutions[:2], feat_dim, init_a, init_b
        )
        self.canonical_decoder = MLP(
            feat_dim,
            num_decoder_layers,
            layer_width=decoder_hidden_size,
            out_dim=3,
            skip_connections=None,
            activation=nn.ReLU(),
            out_activation=None,
        )

        self.deform_planes_encoding = PlaneEncoding(
            resolutions[:2], feat_dim, init_a, init_b
        )

        self.num_temporal_freq = num_temporal_freq
        self.temporal_pos_encoding = TemporalNeRFEncoding(
            1,
            num_temporal_freq,
            freq_min,
            freq_max,
            log_scale=False,
        )

        self.decoder = MLP(
            feat_dim + self.temporal_pos_encoding.get_out_dim(),
            int(num_decoder_layers * 3),
            layer_width=decoder_hidden_size,
            out_dim=self.output_dim,
            skip_connections=(2, 4),
            activation=nn.ReLU(),
            out_activation=None,
            zero_init=zero_init,
        )

    def forward(
        self, inp: Float[Tensor, "*bs 3"]
    ) -> Tuple[Float[Tensor, "*bs 2"], Float[Tensor, "*bs 3"]]:
        # shift to [-1, 1]
        inp_norm = SceneBox.get_normalized_positions(inp, self.aabb) * 2.0 - 1.0

        inp_yx, inp_t = inp_norm[..., 0:2], inp_norm[..., 2:3]

        spatial_feat = self.deform_planes_encoding(inp_yx)

        temporal_enc = self.temporal_pos_encoding(inp_t)
        # [*bs, 2]

        output = self.decoder(
            torch.cat(
                [spatial_feat, temporal_enc.view(-1, temporal_enc.shape[-1])], dim=-1
            )
        )

        canonical_yx = inp_yx.reshape(-1, 2) + output

        ret_rgb_feat = self.canonical_encoding(canonical_yx)
        ret_rgb = self.canonical_decoder(ret_rgb_feat)

        return output, ret_rgb

    def compute_smoothess_loss(
        self,
    ):
        smothness_loss = self.deform_planes_encoding.compute_plane_tv()

        smothness_canonical = self.canonical_encoding.compute_plane_tv()
        return smothness_loss + smothness_canonical

    def get_canonical(
        self, canonical_grid: Float[Tensor, "*bs 2"]
    ) -> Float[Tensor, "*bs 3"]:
        pad_can_grid = torch.cat(
            [canonical_grid, torch.zeros_like(canonical_grid[..., :1])], dim=-1
        )
        pad_can_norm = (
            SceneBox.get_normalized_positions(pad_can_grid, self.aabb) * 2.0 - 1.0
        )

        inp_can_grid = pad_can_norm[..., :2]

        ret_rgb_feat = self.canonical_encoding(inp_can_grid)
        ret_rgb = self.canonical_decoder(ret_rgb_feat)

        return ret_rgb

    def sample_canonical(
        self,
        inp: Float[Tensor, "bs hw 3"],
        canonical_frame: Float[Tensor, "1 H W 3"],
        canonical_grid_yx: Float[Tensor, "bs hw 2"],
    ) -> Float[Tensor, "bs h w 3"]:
        inp_norm = SceneBox.get_normalized_positions(inp, self.aabb) * 2.0 - 1.0

        inp_yx, inp_t = inp_norm[..., 0:2], inp_norm[..., 2:3]
        inp_yx = inp_yx.reshape(-1, 2)
        spatial_feat = self.deform_planes_encoding(inp_yx)

        temporal_enc = self.temporal_pos_encoding(inp_t.view(-1, 1))
        # [*bs, 2]
        output = self.decoder(torch.cat([spatial_feat, temporal_enc], dim=-1))

        canonical_yx = inp_yx + output
        canonical_yx = canonical_yx * 1.1

        can_ymin, can_ymax = (
            canonical_grid_yx[..., 0].min(),
            canonical_grid_yx[..., 0].max(),
        )
        can_xmin, can_xmax = (
            canonical_grid_yx[..., 1].min(),
            canonical_grid_yx[..., 1].max(),
        )
        canonical_yx[..., 0] = (canonical_yx[..., 0] - can_ymin) / (
            can_ymax - can_ymin
        ) * 2.0 - 1.0
        canonical_yx[..., 1] = (canonical_yx[..., 1] - can_xmin) / (
            can_xmax - can_xmin
        ) * 2.0 - 1.0

        canonical_xy = torch.cat(
            [canonical_yx[..., 1:2], canonical_yx[..., 0:1]], dim=-1
        )
        # use grid sample to sample the canonical frame

        # [B, C, H, W]
        canonical_frame = canonical_frame.permute(0, 3, 1, 2).expand(
            inp.shape[0], -1, -1, -1
        )
        H, W = canonical_frame.shape[-2:]
        canonical_xy = canonical_xy.reshape(-1, H, W, 2)

        rec = F.grid_sample(canonical_frame, canonical_xy, align_corners=True)

        rec = rearrange(rec, "b c h w -> b h w c")

        return rec
