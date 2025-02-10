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
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from motionrep.gaussian_3d.scene.gaussian_model import GaussianModel


def render_gaussian(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    cov3D_precomp=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None

    if pipe.compute_cov3D_python or cov3D_precomp is None:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    elif cov3D_precomp is None:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
    # return {"render": rendered_image}


def gaussian_intrin_scale(x_or_y: torch.Tensor, w_or_h: float):

    ret = ((x_or_y + 1.0) * w_or_h - 1.0) * 0.5

    return ret


def render_arrow_in_screen(viewpoint_camera, points_3d):

    # project point motion to 2D using camera:
    w2c = viewpoint_camera.world_view_transform.transpose(0, 1)
    cam_plane_2_img = viewpoint_camera.cam_plane_2_img  # [2, 2]
    cam_plane_2_img = viewpoint_camera.projection_matrix.transpose(0, 1)  # [4, 4]

    full_proj_mat = viewpoint_camera.full_proj_transform

    # [N, 4]
    pts = torch.cat([points_3d, torch.ones_like(points_3d[:, 0:1])], dim=-1)
    # [N, 1, 4] <-  [N, 1, 4] @ [1, 4, 4]
    pts_cam = pts.unsqueeze(-2) @ full_proj_mat.unsqueeze(0)  # [N, 1, 4]

    # start here

    # pts: [N, 4]
    # [1, 4, 4] @ [N, 4, 1] -> [N, 4, 1]
    # from IPython import embed

    # embed()
    # pts_cam = torch.bmm(
    #     full_proj_mat.T.unsqueeze(0), pts.unsqueeze(-1)
    # )  # K*[R,T]*[x,y,z,1]^T to get 2D projection of Gaussians
    # end here
    pts_cam = full_proj_mat.T.unsqueeze(0) @ pts.unsqueeze(-1)

    # print(pts_cam.shape)

    pts_cam = pts_cam.squeeze(-1)  # [N, 4]
    pts_cam = pts_cam[:, :3] / pts_cam[:, 3:]  # [N, 1, 3]

    # print(pts_cam, "after proj")

    pts_cam_yx_pixel = pts_cam[:, :2]
    #  [N, 2] yx => xy
    # pts_cam_xy_pixel = torch.cat(
    #     [pts_cam_xy_pixel[:, [1]], pts_cam_xy_pixel[:, [0]]], dim=-1
    # )

    pts_cam_x, pts_cam_y = pts_cam_yx_pixel[:, 0], pts_cam_yx_pixel[:, 1]

    w, h = viewpoint_camera.image_width, viewpoint_camera.image_height

    pts_cam_x = gaussian_intrin_scale(pts_cam_x, w)
    pts_cam_y = gaussian_intrin_scale(pts_cam_y, h)

    ret_pts_cam_xy = torch.cat(
        [pts_cam_x.unsqueeze(-1), pts_cam_y.unsqueeze(-1)], dim=-1
    )

    # print(ret_pts_cam_xy)

    return ret_pts_cam_xy


def render_arrow_in_screen_back(viewpoint_camera, points_3d):

    # project point motion to 2D using camera:
    w2c = viewpoint_camera.world_view_transform.transpose(0, 1)
    cam_plane_2_img = viewpoint_camera.cam_plane_2_img  # [2, 2]
    cam_plane_2_img = viewpoint_camera.projection_matrix.transpose(0, 1)

    from IPython import embed

    embed()

    R = w2c[:3, :3].unsqueeze(0)  # [1, 3, 3]
    t = w2c[:3, 3].unsqueeze(0)  # [1, 3]

    # [N, 3, 1]
    pts = torch.cat([points_3d, torch.ones_like(points_3d[:, 0:1])], dim=-1)
    pts_cam = w2c.unsqueeze(0) @ pts.unsqueeze(-1)  # [N, 4, 1]
    # pts_cam = R @ (pc._xyz.unsqueeze(-1)) + t[:, None]
    depth = pts_cam[:, 2, 0]  # [N]
    # print("depth", depth.shape, depth.max(), depth.mean(), depth.min())

    # [N, 2]
    pts_cam_xy = pts_cam[:, :2, 0] / depth.unsqueeze(-1)

    pts_cam_xy_pixel = cam_plane_2_img.unsqueeze(0) @ pts_cam_xy.unsqueeze(
        -1
    )  # [N, 2, 1]
    pts_cam_xy_pixel = pts_cam_xy_pixel.squeeze(-1)  # [N, 2]

    #  [N, 2] yx => xy
    pts_cam_xy_pixel = torch.cat(
        [pts_cam_xy_pixel[:, [1]], pts_cam_xy_pixel[:, [0]]], dim=-1
    )

    return pts_cam_xy_pixel


# for spherecal harmonics


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5
