import torch
from motionrep.gaussian_3d.scene.gaussian_model import GaussianModel
import math

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from typing import Callable


def render_flow_depth_w_gaussian(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    point_disp: torch.Tensor,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    Args:
        point_disp: [N, 3]
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
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    # project point motion to 2D using camera:
    w2c = viewpoint_camera.world_view_transform.transpose(0, 1)
    cam_plane_2_img = viewpoint_camera.cam_plane_2_img  # [2, 2]

    R = w2c[:3, :3].unsqueeze(0)  # [1, 3, 3]
    t = w2c[:3, 3].unsqueeze(0)  # [1, 3]

    # [N, 3, 1]
    pts = torch.cat([pc._xyz, torch.ones_like(pc._xyz[:, 0:1])], dim=-1)
    pts_cam = w2c.unsqueeze(0) @ pts.unsqueeze(-1)  # [N, 4, 1]
    # pts_cam = R @ (pc._xyz.unsqueeze(-1)) + t[:, None]
    depth = pts_cam[:, 2, 0]  # [N]
    # print("depth", depth.shape, depth.max(), depth.mean(), depth.min())

    point_disp_pad = torch.cat(
        [point_disp, torch.zeros_like(point_disp[:, 0:1])], dim=-1
    )  # [N, 4]

    pts_motion = w2c.unsqueeze(0) @ point_disp_pad.unsqueeze(-1)  # [N, 4, 1]

    # [N, 2]
    pts_motion_xy = pts_motion[:, :2, 0] / depth.unsqueeze(-1)

    pts_motion_xy_pixel = cam_plane_2_img.unsqueeze(0) @ pts_motion_xy.unsqueeze(
        -1
    )  # [N, 2, 1]
    pts_motion_xy_pixel = pts_motion_xy_pixel.squeeze(-1)  # [N, 2]

    colors_precomp = torch.cat(
        [pts_motion_xy_pixel, depth.unsqueeze(dim=-1)], dim=-1
    )  # [N, 3]

    # print("converted 2D motion precompute: ", colors_precomp.shape, shs, colors_precomp.max(), colors_precomp.min(), colors_precomp.mean())
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

    # return {
    #     "render": rendered_image,
    #     "viewspace_points": screenspace_points,
    #     "visibility_filter": radii > 0,
    #     "radii": radii,
    # }

    return {"render": rendered_image}
