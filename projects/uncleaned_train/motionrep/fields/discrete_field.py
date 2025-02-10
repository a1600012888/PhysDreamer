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


class PointSetMotionSE3(nn.Module):
    """Temporal Kplanes SE(3) fields.

    Args:
        aabb: axis-aligned bounding box.
            aabb[0] is the minimum (x,y,z) point.
            aabb[1] is the maximum (x,y,z) point.
        resolutions: resolutions of the kplanes. in an order of [x, y, z ,t].

    """

    def __init__(
        self,
        inp_x: Float[Tensor, "*bs 3"],
        aabb: Float[Tensor, "2 3"],
        rotation_type: Literal["quaternion", "6d"] = "6d",
        num_frames: int = 20,
        distance_lamba=100.0,
        topk_nn: int = 20,  # the same neighboor size as dynamic gaussian
    ):
        super().__init__()

        self.register_buffer("aabb", aabb)
        output_dim_dict = {"quaternion": 4 + 3, "6d": 6 + 3}
        self.output_dim = output_dim_dict[rotation_type]
        self.rotation_type = rotation_type

        self.register_buffer("inp_x", inp_x.detach())

        self.num_frames = num_frames

        # init parameters:
        translation = nn.Parameter(
            torch.zeros(num_frames + 1, inp_x.shape[0], 3).requires_grad_(True)
        )
        rotation = nn.Parameter(
            torch.ones(
                (num_frames + 1, inp_x.shape[0], self.output_dim - 3)
            ).requires_grad_(True)
        )
        self.register_parameter("translation", translation)
        self.register_parameter("rotation", rotation)

        # [num_points, topk]
        print(inp_x.shape, "input shape gaussian")
        knn_dist, knn_ind = self.construct_knn(inp_x, topk=topk_nn)

        # [num_points, topk]
        self.distance_weight = torch.exp(-1.0 * distance_lamba * knn_dist)
        self.knn_index = knn_ind  # torch.long

        self.precompute_isometry = self.prepare_isometry(inp_x, knn_ind)

        self.inp_time_list = []

    def construct_knn(self, inpx: Float[Tensor, "*bs 3"], topk=10, chunk_size=5000):
        # compute topk nearest neighbors for each point, and the distance

        knn_dist_list, knn_ind_list = [], []
        num_step = inpx.shape[0] // chunk_size + 1

        with torch.no_grad():
            for i in range(num_step):
                end_ind = min((i + 1) * chunk_size, inpx.shape[0])

                src_points = inpx[i * chunk_size : end_ind]
                # compute the distance matrix
                cdist = torch.cdist(src_points, inpx)

                print(cdist.shape, "cdist")
                # get the topk nearest neighbors
                knn_dist, knn_ind = torch.topk(cdist, topk, dim=1, largest=False)
                knn_dist_list.append(knn_dist)
                knn_ind_list.append(knn_ind)

            knn_dist = torch.cat(knn_dist_list, dim=0)
            knn_ind = torch.cat(knn_ind_list, dim=0)
        return knn_dist, knn_ind

    def prepare_isometry(self, points, knn_ind):
        # [num_points, topk, 3]
        p_nn = points[knn_ind]

        dsp = points[:, None, :] - p_nn

        distance = torch.norm(dsp, dim=-1)

        # [num_points, topk]
        return distance

    def _forward_single_time(self, time_ind: int):
        if self.rotation_type == "6d":
            rotation_6d, translation = (
                self.rotation[time_ind],
                self.translation[time_ind],
            )
            R_mat = rotation_6d_to_matrix(rotation_6d)

        elif self.rotation_type == "quaternion":
            quat, translation = self.rotation[time_ind], self.translation[time_ind]

            quat = torch.tanh(quat)
            R_mat = quaternion_to_matrix(quat)

        return R_mat, translation

    def forward(
        self,
        inp: Float[Tensor, "*bs 4"],
        **kwargs,
    ) -> Tuple[Float[Tensor, "*bs 3 3"], Float[Tensor, "*bs 3"]]:
        inpx, inpt = inp[:, :3], inp[:, 3:]

        time_ind = torch.round(inpt * (self.num_frames)).long()[0].item()
        R_mat, translation = self._forward_single_time(time_ind)

        self.inp_time_list.append(time_ind)
        if len(self.inp_time_list) > 20:
            self.inp_time_list.pop(0)

        return R_mat, translation

    def compute_smoothess_loss(
        self,
    ):
        # temporal_smoothness_loss = torch.tensor([0.0]).cuda()
        temporal_smoothness_loss = self.compute_isometry_loss()
        smothness_loss = self.compute_arap_loss()

        return temporal_smoothness_loss, smothness_loss

    def compute_arap_loss(
        self,
    ):
        arap_loss = 0.0

        # random sample 16 frames
        random_frame_ind_list = torch.randint(0, self.num_frames - 1, (16,))

        for i in self.inp_time_list:
            r1, t1 = self._forward_single_time(i)
            r2, t2 = self._forward_single_time(i + 1)

            # [num_points, topk, 3, 3], [num_points, topk, 3]
            r1_nn, t1_nn = r1[self.knn_index], t1[self.knn_index]
            r2_nn, t2_nn = r2[self.knn_index], t2[self.knn_index]

            # displacement between neighboor points
            #   shape of [num_points, topk, 3]
            dsp_t0 = t1_nn - t1[:, None, :]
            dsp_t1 = t2_nn - t2[:, None, :]

            # rotation matrix from frame-1 to frame-0

            r_mat_1to0 = torch.bmm(r1, r2.transpose(1, 2))  # [N, 3, 3]
            # [N, 3, 3] => [N, topk, 3, 3]
            r_mat_1to0 = r_mat_1to0.unsqueeze(1).repeat(
                1, self.knn_index.shape[1], 1, 1
            )
            dsp_t1_to_0 = torch.matmul(r_mat_1to0, dsp_t1[:, :, :, None]).squeeze(-1)
            # compute the arap loss
            arap_loss += torch.mean(
                torch.norm(dsp_t0 - dsp_t1_to_0, dim=-1) * self.distance_weight
            )
        return arap_loss

    def compute_isometry_loss(
        self,
    ):
        iso_loss = 0.0
        # random sample 16 frames
        random_frame_ind_list = torch.randint(0, self.num_frames - 1, (16,))

        for i in self.inp_time_list:
            r1, t1 = self._forward_single_time(i)
            points = self.inp_x + t1
            distance_mat = self.prepare_isometry(points, self.knn_index)

            iso_loss += torch.mean(
                torch.abs(distance_mat - self.precompute_isometry)
                * self.distance_weight
            )
        return iso_loss

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
