import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
from typing import Optional, Sequence, Tuple, List
from motionrep.losses.smoothness_loss import compute_plane_smoothness, compute_plane_tv


class TemporalKplanesEncoding(nn.Module):
    """

    Args:
        resolutions (Sequence[int]): xyzt resolutions.
    """

    def __init__(
        self,
        resolutions: Sequence[int],
        feat_dim: int = 32,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce="sum",  # Literal["sum", "product", "cat"] = "sum",
    ):
        super().__init__()

        self.resolutions = resolutions

        if reduce == "cat":
            feat_dim = feat_dim // 3
        self.feat_dim = feat_dim

        self.reduce = reduce

        self.in_dim = 4

        self.plane_coefs = nn.ParameterList()

        self.coo_combs = [[0, 3], [1, 3], [2, 3]]
        # [(x, t), (y, t), (z, t)]
        for coo_comb in self.coo_combs:
            # [feat_dim, time_resolution, spatial_resolution]
            new_plane_coef = nn.Parameter(
                torch.empty(
                    [
                        self.feat_dim,
                        resolutions[coo_comb[1]],
                        resolutions[coo_comb[0]],  # flip?
                    ]
                )
            )

            # when init to ones?

            nn.init.uniform_(new_plane_coef, a=init_a, b=init_b)
            self.plane_coefs.append(new_plane_coef)

    def forward(self, inp: Float[Tensor, "*bs 4"]):
        output = 1.0 if self.reduce == "product" else 0.0
        if self.reduce == "cat":
            output = []
        for ci, coo_comb in enumerate(self.coo_combs):
            grid = self.plane_coefs[ci].unsqueeze(0)  # [1, feature_dim, reso1, reso2]
            coords = inp[..., coo_comb].view(1, 1, -1, 2)  # [1, 1, flattened_bs, 2]

            interp = F.grid_sample(
                grid, coords, align_corners=True, padding_mode="border"
            )  # [1, output_dim, 1, flattened_bs]
            interp = interp.view(self.feat_dim, -1).T  # [flattened_bs, output_dim]

            if self.reduce == "product":
                output = output * interp
            elif self.reduce == "sum":
                output = output + interp
            elif self.reduce == "cat":
                output.append(interp)

        if self.reduce == "cat":
            # [flattened_bs, output_dim * 3]
            output = torch.cat(output, dim=-1)

        return output

    def compute_temporal_smoothness(
        self,
    ):
        ret_loss = 0.0

        for plane_coef in self.plane_coefs:
            ret_loss += compute_plane_smoothness(plane_coef)

        return ret_loss

    def compute_plane_tv(
        self,
    ):
        ret_loss = 0.0

        for plane_coef in self.plane_coefs:
            ret_loss += compute_plane_tv(plane_coef)

        return ret_loss

    def visualize(
        self,
    ) -> Tuple[Float[Tensor, "3 H W"]]:
        """Visualize the encoding as a RGB images

        Returns:
            Tuple[Float[Tensor, "3 H W"]]
        """
        pass

    @staticmethod
    def functional_forward(
        plane_coefs: List[Float[Tensor, "feat_dim H W"]],
        inp: Float[Tensor, "*bs 4"],
        reduce: str = "sum",
        coo_combs: Optional[List[List[int]]] = [[0, 3], [1, 3], [2, 3]],
    ):
        assert reduce in ["sum", "product", "cat"]
        output = 1.0 if reduce == "product" else 0.0

        if reduce == "cat":
            output = []
        for ci, coo_comb in enumerate(coo_combs):
            grid = plane_coefs[ci].unsqueeze(0)  # [1, feature_dim, reso1, reso2]
            feat_dim = grid.shape[1]
            coords = inp[..., coo_comb].view(1, 1, -1, 2)  # [1, 1, flattened_bs, 2]

            interp = F.grid_sample(
                grid, coords, align_corners=True, padding_mode="border"
            )  # [1, output_dim, 1, flattened_bs]
            interp = interp.view(feat_dim, -1).T  # [flattened_bs, output_dim]

            if reduce == "product":
                output = output * interp
            elif reduce == "sum":
                output = output + interp
            elif reduce == "cat":
                output.append(interp)

        if reduce == "cat":
            # [flattened_bs, output_dim * 3]
            output = torch.cat(output, dim=-1)

        return output


class TriplanesEncoding(nn.Module):
    """

    Args:
        resolutions (Sequence[int]): xyz resolutions.
    """

    def __init__(
        self,
        resolutions: Sequence[int],
        feat_dim: int = 32,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce="sum",  # Literal["sum", "product", "cat"] = "sum",
    ):
        super().__init__()

        self.resolutions = resolutions

        if reduce == "cat":
            feat_dim = feat_dim#  // 3
        self.feat_dim = feat_dim

        self.reduce = reduce

        self.in_dim = 3

        self.plane_coefs = nn.ParameterList()

        self.coo_combs = [[0, 1], [0, 2], [1, 2]]
        # [(x, t), (y, t), (z, t)]
        for coo_comb in self.coo_combs:
            new_plane_coef = nn.Parameter(
                torch.empty(
                    [
                        self.feat_dim,
                        resolutions[coo_comb[1]],
                        resolutions[coo_comb[0]],
                    ]
                )
            )

            # when init to ones?

            nn.init.uniform_(new_plane_coef, a=init_a, b=init_b)
            self.plane_coefs.append(new_plane_coef)

    def forward(self, inp: Float[Tensor, "*bs 3"]):
        output = 1.0 if self.reduce == "product" else 0.0
        if self.reduce == "cat":
            output = []
        for ci, coo_comb in enumerate(self.coo_combs):
            grid = self.plane_coefs[ci].unsqueeze(0)  # [1, feature_dim, reso1, reso2]
            coords = inp[..., coo_comb].view(1, 1, -1, 2)  # [1, 1, flattened_bs, 2]

            interp = F.grid_sample(
                grid, coords, align_corners=True, padding_mode="border"
            )  # [1, output_dim, 1, flattened_bs]
            interp = interp.view(self.feat_dim, -1).T  # [flattened_bs, output_dim]

            if self.reduce == "product":
                output = output * interp
            elif self.reduce == "sum":
                output = output + interp
            elif self.reduce == "cat":
                output.append(interp)

        if self.reduce == "cat":
            # [flattened_bs, output_dim * 3]
            output = torch.cat(output, dim=-1)

        return output

    def compute_plane_tv(
        self,
    ):
        ret_loss = 0.0

        for plane_coef in self.plane_coefs:
            ret_loss += compute_plane_tv(plane_coef)

        return ret_loss


class PlaneEncoding(nn.Module):
    """

    Args:
        resolutions (Sequence[int]): xyz resolutions.
    """

    def __init__(
        self,
        resolutions: Sequence[int],  # [y_res, x_res]
        feat_dim: int = 32,
        init_a: float = 0.1,
        init_b: float = 0.5,
    ):
        super().__init__()

        self.resolutions = resolutions

        self.feat_dim = feat_dim
        self.in_dim = 2

        self.plane_coefs = nn.ParameterList()

        self.coo_combs = [[0, 1]]
        for coo_comb in self.coo_combs:
            new_plane_coef = nn.Parameter(
                torch.empty(
                    [
                        self.feat_dim,
                        resolutions[coo_comb[1]],
                        resolutions[coo_comb[0]],
                    ]
                )
            )

            # when init to ones?

            nn.init.uniform_(new_plane_coef, a=init_a, b=init_b)
            self.plane_coefs.append(new_plane_coef)

    def forward(self, inp: Float[Tensor, "*bs 2"]):

        for ci, coo_comb in enumerate(self.coo_combs):
            grid = self.plane_coefs[ci].unsqueeze(0)  # [1, feature_dim, reso1, reso2]
            coords = inp[..., coo_comb].view(1, 1, -1, 2)  # [1, 1, flattened_bs, 2]

            interp = F.grid_sample(
                grid, coords, align_corners=True, padding_mode="border"
            )  # [1, output_dim, 1, flattened_bs]
            interp = interp.view(self.feat_dim, -1).T  # [flattened_bs, output_dim]

            output = interp

        return output

    def compute_plane_tv(
        self,
    ):
        ret_loss = 0.0

        for plane_coef in self.plane_coefs:
            ret_loss += compute_plane_tv(plane_coef)

        return ret_loss


class TemporalNeRFEncoding(nn.Module):
    def __init__(
        self,
        in_dim,  # : int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        log_scale: bool = False,
        include_input: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.log_scale = log_scale
        self.include_input = include_input

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
    
        # freqs = 2 ** torch.linspace(
        freqs = torch.linspace(
            self.min_freq, self.max_freq, self.num_frequencies, device=in_tensor.device
        )
        if self.log_scale:
            freqs = 2 ** freqs
        scaled_inputs = (
            scaled_in_tensor[..., None] * freqs
        )  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(
            *scaled_inputs.shape[:-2], -1
        )  # [..., "input_dim" * "num_scales"]

        encoded_inputs = torch.sin(
            torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1)
        )
        return encoded_inputs
