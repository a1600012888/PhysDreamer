#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from physdreamer.gaussian_3d.utils.general_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
)
from torch import nn
import os
from physdreamer.gaussian_3d.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from physdreamer.gaussian_3d.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from physdreamer.gaussian_3d.utils.graphics_utils import BasicPointCloud
from physdreamer.gaussian_3d.utils.general_utils import (
    strip_symmetric,
    build_scaling_rotation,
)
from physdreamer.gaussian_3d.utils.rigid_body_utils import (
    get_rigid_transform,
    matrix_to_quaternion,
    quaternion_multiply,
)


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int = 3):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.matched_inds = None

    def capture(self):
        if self.optimizer is None:
            optim_state = None
        else:
            optim_state = self.optimizer.state_dict()

        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            optim_state,
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args

        if training_args is not None:
            self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict)

    def capture_training_args(
        self,
    ):
        pass

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        # typo here?
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def apply_discrete_offset_filds(self, origin_points, offsets):
        """
        Args:
            origin_points: (N_r, 3)
            offsets: (N_r, 3)
        """

        # since origin points and self._xyz might not be matched, we need to first
        #   compute the distance between origin points and self._xyz
        #   then find the nearest point in self._xyz for each origin point

        # compute the distance between origin points and self._xyz
        # [N_r, num_points]
        dist = torch.cdist(origin_points, self._xyz)
        # find the nearest point in self._xyz for each origin point
        _, idx = torch.min(dist, dim=0)

        # apply offsets

        new_xyz = self._xyz + offsets[idx]

        if self.optimizer is None:
            optim_state = None
        else:
            optim_state = self.optimizer.state_dict()

        new_model_args = (
            self.active_sh_degree,
            new_xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            optim_state,
            self.spatial_lr_scale,
        )

        ret_gaussian = GaussianModel(self.max_sh_degree)
        ret_gaussian.restore(new_model_args, None)

        return ret_gaussian

    def apply_discrete_offset_filds_with_R(self, origin_points, offsets, topk=6):
        """
        Args:
            origin_points: (N_r, 3)
            offsets: (N_r, 3)
        """

        # since origin points and self._xyz might not be matched, we need to first
        #   compute the distance between origin points and self._xyz
        #   then find the nearest point in self._xyz for each origin point

        if self.matched_inds is None:
            # compute the distance between origin points and self._xyz
            # [N_r, num_points]
            dist = torch.cdist(origin_points, self._xyz) * -1.0
            # find the nearest point in self._xyz for each origin point

            # idxs: [topk, num_points]
            print(dist.shape, topk, dist[0])
            _, idxs = torch.topk(dist, topk, dim=0)

            self.matched_inds = idxs
        else:
            idxs = self.matched_inds

        # [topk, num_points, 3] => [num_points, topk, 3]
        matched_topk_offsets = offsets[idxs].transpose(0, 1)
        source_points = origin_points[idxs].transpose(0, 1)

        # [num_points, 3, 3/1]
        R, t = get_rigid_transform(source_points, source_points + matched_topk_offsets)

        # new_xyz = R @ self._xyz.unsqueeze(dim=-1) + t
        # new_xyz = new_xyz.squeeze(dim=-1)

        avg_offsets = matched_topk_offsets.mean(dim=1)
        new_xyz = self._xyz + avg_offsets  # offset directly

        new_rotation = quaternion_multiply(matrix_to_quaternion(R), self._rotation)

        if self.optimizer is None:
            optim_state = None
        else:
            optim_state = self.optimizer.state_dict()

        new_model_args = (
            self.active_sh_degree,
            new_xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            new_rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            optim_state,
            self.spatial_lr_scale,
        )

        ret_gaussian = GaussianModel(self.max_sh_degree)
        ret_gaussian.restore(new_model_args, None)

        return ret_gaussian

    def apply_se3_fields(
        self,
        se3_model,
        timestamp: float,
        freeze_mask=None,
    ):
        """
        Args:
            se3_model: SE3Model
            timestamp: float.  in range [0, 1]
            freeze_mask: [N]
        """

        inp_time = torch.ones_like(self._xyz[:, 0:1]) * timestamp
        inp = torch.cat([self._xyz, inp_time], dim=-1)

        if freeze_mask is not None:
            moving_mask = torch.logical_not(freeze_mask)
            inp = inp[moving_mask, ...]
        # [bs, 3, 3]. [bs, 3]
        R, t = se3_model(inp)

        # print("abs t mean", torch.abs(t).mean(dim=0))
        # new_xyz = (R @ self._xyz.unsqueeze(dim=-1)).squeeze(dim=-1) + t

        if freeze_mask is None:
            new_xyz = self._xyz + t
            new_rotation = quaternion_multiply(matrix_to_quaternion(R), self._rotation)
        else:
            new_xyz = self._xyz.clone()
            new_xyz[moving_mask, ...] += t
            new_rotation = self._rotation.clone()
            new_rotation[moving_mask, ...] = quaternion_multiply(
                matrix_to_quaternion(R), self._rotation[moving_mask, ...]
            )

        if self.optimizer is None:
            optim_state = None
        else:
            optim_state = self.optimizer.state_dict()

        new_model_args = (
            self.active_sh_degree,
            new_xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            new_rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            optim_state,
            self.spatial_lr_scale,
        )

        ret_gaussian = GaussianModel(self.max_sh_degree)
        ret_gaussian.restore(new_model_args, None)

        return ret_gaussian

    def apply_offset_fields(self, offset_field, timestamp: float):
        """
        Args:
            se3_model: SE3Model
            timestamp: float.  in range [0, 1]
        """

        inp_time = torch.ones_like(self._xyz[:, 0:1]) * timestamp
        inp = torch.cat([self._xyz, inp_time], dim=-1)
        # [bs, 3, 3]. [bs, 3]
        offsets = offset_field(inp)

        # print("abs t mean", torch.abs(t).mean(dim=0))
        new_xyz = self._xyz + offsets

        if self.optimizer is None:
            optim_state = None
        else:
            optim_state = self.optimizer.state_dict()

        new_model_args = (
            self.active_sh_degree,
            new_xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            optim_state,
            self.spatial_lr_scale,
        )

        ret_gaussian = GaussianModel(self.max_sh_degree)
        ret_gaussian.restore(new_model_args, None)

        return ret_gaussian

    def apply_offset_fields_with_R(self, offset_field, timestamp: float, eps=1e-2):
        """
        Args:
            se3_model: SE3Model
            timestamp: float.  in range [0, 1]
        """

        # [4, 3]
        inp_perterb = (
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],  # add this will coplanar?
                    [+eps, -eps, -eps],
                    [-eps, -eps, +eps],
                    [-eps, +eps, -eps],
                    [+eps, +eps, +eps],
                ],
            )
            .to(self._xyz.device)
            .float()
        )
        #  => [N, 4, 3]
        source_points = self._xyz.unsqueeze(dim=1) + inp_perterb.unsqueeze(dim=0)
        num_points = source_points.shape[0]

        inpx = source_points.flatten(end_dim=1)
        inp_time = torch.ones_like(inpx[:, 0:1]) * timestamp

        inp = torch.cat([inpx, inp_time], dim=-1)

        sampled_offsets = offset_field(inp).reshape((num_points, -1, 3))

        R, t = get_rigid_transform(source_points, source_points + sampled_offsets)

        # new_xyz = R @ self._xyz.unsqueeze(dim=-1) + t
        # new_xyz = new_xyz.squeeze(dim=-1)

        avg_offsets = sampled_offsets.mean(dim=1)
        new_xyz = self._xyz + avg_offsets  # offset directly

        new_rotation = quaternion_multiply(matrix_to_quaternion(R), self._rotation)

        if self.optimizer is None:
            optim_state = None
        else:
            optim_state = self.optimizer.state_dict()

        new_model_args = (
            self.active_sh_degree,
            new_xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            new_rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            optim_state,
            self.spatial_lr_scale,
        )

        ret_gaussian = GaussianModel(self.max_sh_degree)
        ret_gaussian.restore(new_model_args, None)

        return ret_gaussian

    def init_from_mesh(
        self,
        mesh_path: str,
        num_gaussians: int = 10000,
    ):
        import point_cloud_utils as pcu

        mesh = pcu.load_triangle_mesh(mesh_path)

        v, f = mesh.v, mesh.f

        v_n = pcu.estimate_mesh_normals(v, f)
        vert_colors = mesh.vertex_data.colors

        fid, bc = pcu.sample_mesh_random(v, f, num_gaussians)

        # Interpolate the vertex positions and normals using the returned barycentric coordinates
        # to get sample positions and normals
        rand_positions = pcu.interpolate_barycentric_coords(f, fid, bc, v)
        rand_normals = pcu.interpolate_barycentric_coords(f, fid, bc, v_n)
        rand_colors = pcu.interpolate_barycentric_coords(f, fid, bc, vert_colors)[:, :3]

        # copy original pointcloud init functions

        fused_point_cloud = torch.tensor(np.asarray(rand_positions)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(rand_colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        # typo here?
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(rand_positions)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def detach_grad(
        self,
    ):
        self._xyz.requires_grad = False
        self._features_dc.requires_grad = False
        self._features_rest.requires_grad = False
        self._scaling.requires_grad = False
        self._rotation.requires_grad = False
        self._opacity.requires_grad = False

    def apply_mask(self, mask):
        new_xyz = self._xyz[mask]
        if self.xyz_gradient_accum.shape == self._xyz.shape:
            new_xyz_gradient_accum = self.xyz_gradient_accum[mask]
            new_denom = self.denom[mask]
        else:
            new_xyz_gradient_accum = self.xyz_gradient_accum
            new_denom = self.denom
        new_model_args = (
            self.active_sh_degree,
            new_xyz,
            self._features_dc[mask],
            self._features_rest[mask],
            self._scaling[mask],
            self._rotation[mask],
            self._opacity[mask],
            self.max_radii2D,
            new_xyz_gradient_accum,
            new_denom,
            None,
            self.spatial_lr_scale,
        )

        ret_gaussian = GaussianModel(self.max_sh_degree)
        ret_gaussian.restore(new_model_args, None)

        return ret_gaussian

    @torch.no_grad()
    def extract_fields(self, resolution=128, num_blocks=16, relax_ratio=1.5):
        # resolution: resolution of field

        block_size = 2 / num_blocks

        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = self.get_opacity

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)

        opacities = opacities[mask]
        xyzs = self.get_xyz[mask]
        stds = self.get_scaling[mask]

        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.center = (mn + mx) / 2
        self.scale = 1.0 / (mx - mn).amax().item()

        print("gaussian center, scale", self.center, self.scale)
        xyzs = (xyzs - self.center) * self.scale
        stds = stds * self.scale

        covs = self.covariance_activation(stds, 1, self._rotation[mask])

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)

        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3]
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                        dim=-1,
                    ).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask]  # [L, 3]
                    mask_covs = covs[mask]  # [L, 6]
                    mask_opas = opacities[mask].view(1, -1)  # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(
                        1, mask_covs.shape[0], 1
                    ) - mask_xyzs.unsqueeze(
                        0
                    )  # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(
                        pts.shape[0], 1, 1
                    )  # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(
                            g_pts[:, start:end].reshape(-1, 3),
                            g_covs[:, start:end].reshape(-1, 6),
                        ).reshape(
                            pts.shape[0], -1
                        )  # [M, l]
                        val += (mask_opas[:, start:end] * w).sum(-1)

                    # kiui.lo(val, mask_opas, w)

                    occ[
                        xi * split_size : xi * split_size + len(xs),
                        yi * split_size : yi * split_size + len(ys),
                        zi * split_size : zi * split_size + len(zs),
                    ] = val.reshape(len(xs), len(ys), len(zs))

        return occ

    def extract_mesh(self, path, density_thresh=1, resolution=128, decimate_target=1e5):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        from physdreamer.gaussian_3d.scene.mesh import Mesh
        from physdreamer.gaussian_3d.scene.mesh_utils import decimate_mesh, clean_mesh

        occ = self.extract_fields(resolution).detach().cpu().numpy()

        print(occ.shape, occ.min(), occ.max(), occ.mean(), "occ stats")
        print(np.percentile(occ, [0, 1, 5, 10, 50, 90, 95, 99, 100]), "occ percentiles")
        import mcubes

        vertices, triangles = mcubes.marching_cubes(occ, density_thresh)
        vertices = vertices / (resolution - 1.0) * 2 - 1

        # transform back to the original space
        vertices = vertices / self.scale + self.center.detach().cpu().numpy()

        vertices, triangles = clean_mesh(
            vertices, triangles, remesh=True, remesh_size=0.015
        )
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

        v = torch.from_numpy(vertices.astype(np.float32)).contiguous().cuda()
        f = torch.from_numpy(triangles.astype(np.int32)).contiguous().cuda()

        print(
            f"[INFO] marching cubes result: {v.shape} ({v.min().item()}-{v.max().item()}), {f.shape}"
        )

        mesh = Mesh(v=v, f=f, device="cuda")

        return mesh


def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = (
        covs[:, 0],
        covs[:, 1],
        covs[:, 2],
        covs[:, 3],
        covs[:, 4],
        covs[:, 5],
    )

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = (
        -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f)
        - x * y * inv_b
        - x * z * inv_c
        - y * z * inv_e
    )

    power[power > 0] = -1e10  # abnormal values... make weights 0

    return torch.exp(power)
