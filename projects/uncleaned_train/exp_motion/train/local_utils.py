import os
import torch
from jaxtyping import Float, Int, Shaped
from torch import Tensor
from time import time
from omegaconf import OmegaConf
from motionrep.fields.se3_field import TemporalKplanesSE3fields
from motionrep.fields.triplane_field import TriplaneFields, TriplaneFieldsWithEntropy
from motionrep.utils.svd_helpper import load_model_from_config

from motionrep.gaussian_3d.gaussian_renderer.render import (
    render_gaussian,
    render_arrow_in_screen,
)
from motionrep.gaussian_3d.gaussian_renderer.flow_depth_render import (
    render_flow_depth_w_gaussian,
)
import cv2
import numpy as np
from sklearn.cluster import KMeans
from time import time

from motionrep.gaussian_3d.utils.rigid_body_utils import (
    get_rigid_transform,
    matrix_to_quaternion,
    quaternion_multiply,
)


def cycle(dl: torch.utils.data.DataLoader):
    while True:
        for data in dl:
            yield data


def load_motion_model(model, checkpoint_path):
    model_path = os.path.join(checkpoint_path, "model.pt")
    model.load_state_dict(torch.load(model_path))
    print("load model from: ", model_path)
    return model


def create_spatial_fields(
    args, output_dim, aabb: Float[Tensor, "2 3"], add_entropy=True
):

    sp_res = args.sim_res

    resolutions = [sp_res, sp_res, sp_res]
    reduce = "sum"

    if args.entropy_cls > 0 and add_entropy:
        model = TriplaneFieldsWithEntropy(
            aabb,
            resolutions,
            feat_dim=32,
            init_a=0.1,
            init_b=0.5,
            reduce=reduce,
            num_decoder_layers=2,
            decoder_hidden_size=32,
            output_dim=output_dim,
            zero_init=args.zero_init,
            num_cls=args.entropy_cls,
        )
    else:
        model = TriplaneFields(
            aabb,
            resolutions,
            feat_dim=32,
            init_a=0.1,
            init_b=0.5,
            reduce=reduce,
            num_decoder_layers=2,
            decoder_hidden_size=32,
            output_dim=output_dim,
            zero_init=args.zero_init,
        )
    if args.zero_init:
        print("=> zero init the last layer for Spatial MLP")

    return model


def create_motion_model(
    args,
    aabb: Float[Tensor, "2 3"],
    num_frames=None,
):
    assert args.model in ["se3_field"]

    sp_res = args.spatial_res
    if num_frames is None:
        num_frames = args.num_frames
    resolutions = [sp_res, sp_res, sp_res, (num_frames) // 2 + 1]
    # resolutions = [64, 64, 64, num_frames // 2 + 1]
    reduce = "sum"

    model = TemporalKplanesSE3fields(
        aabb,
        resolutions,
        feat_dim=args.feat_dim,
        init_a=0.1,
        init_b=0.5,
        reduce=reduce,
        num_decoder_layers=args.num_decoder_layers,
        decoder_hidden_size=args.decoder_hidden_size,
        zero_init=args.zero_init,
    )
    if args.zero_init:
        print("=> zero init the last layer for MLP")

    return model


def create_velocity_model(
    args,
    aabb: Float[Tensor, "2 3"],
):

    from motionrep.fields.offset_field import TemporalKplanesOffsetfields

    sp_res = args.sim_res
    resolutions = [sp_res, sp_res, sp_res, (args.num_frames) // 2 + 1]
    reduce = "sum"
    model = TemporalKplanesOffsetfields(
        aabb,
        resolutions,
        feat_dim=32,
        init_a=0.1,
        init_b=0.5,
        reduce=reduce,
        num_decoder_layers=2,
        decoder_hidden_size=32,
        zero_init=args.zero_init,
    )
    if args.zero_init:
        print("=> zero init the last layer for velocity MLP")
    return model


def create_svd_model(model_name="svd_full", ckpt_path=None):
    state = dict()
    cfg_path_dict = {
        "svd_full": "svd_configs/svd_full_decoder.yaml",
    }
    config = cfg_path_dict[model_name]

    config = OmegaConf.load(config)

    if ckpt_path is not None:
        # overwrite config.
        config.model.params.ckpt_path = ckpt_path

    s_time = time()
    # model will automatically load when create
    model, msg = load_model_from_config(config, None)

    state["config"] = config

    print(f"Loading svd model takes {time() - s_time} seconds")

    return model, state


class LinearStepAnneal(object):
    # def __init__(self, total_iters, start_state=[0.02, 0.98], end_state=[0.50, 0.98]):
    def __init__(
        self,
        total_iters,
        start_state=[0.02, 0.98],
        end_state=[0.02, 0.98],
        plateau_iters=-1,
        warmup_step=300,
    ):
        self.total_iters = total_iters

        if plateau_iters < 0:
            plateau_iters = int(total_iters * 0.2)

        if warmup_step <= 0:
            warmup_step = 0

        self.total_iters = max(total_iters - plateau_iters - warmup_step, 10)

        self.start_state = start_state
        self.end_state = end_state
        self.warmup_step = warmup_step

    def compute_state(self, cur_iter):

        if self.warmup_step > 0:
            cur_iter = max(0, cur_iter - self.warmup_step)
        if cur_iter >= self.total_iters:
            return self.end_state
        ret = []
        for s, e in zip(self.start_state, self.end_state):
            ret.append(s + (e - s) * cur_iter / self.total_iters)
        return ret


def setup_boundary_condition(
    xyzs_over_time: torch.Tensor, mpm_solver, mpm_state, num_filled=0
):

    init_velocity = xyzs_over_time[1] - xyzs_over_time[0]
    init_velocity_mag = torch.norm(init_velocity, dim=-1)

    # 10% of the velocity
    velocity_thres = torch.quantile(init_velocity_mag, 0.1, dim=0)

    # [n_particles]. 1 for freeze, 0 for moving
    freeze_mask = init_velocity_mag < velocity_thres
    freeze_mask = freeze_mask.type(torch.int)
    if num_filled > 0:
        freeze_mask = torch.cat(
            [freeze_mask, freeze_mask.new_zeros(num_filled).type(torch.int)], dim=0
        )
    num_freeze_pts = freeze_mask.sum()
    print("num freeze pts from static points", num_freeze_pts.item())

    free_velocity = torch.zeros_like(init_velocity[0])  # [3] in device

    mpm_solver.enforce_particle_velocity_by_mask(
        mpm_state, freeze_mask, free_velocity, start_time=0, end_time=100000
    )

    return freeze_mask


def setup_plannar_boundary_condition(
    xyzs_over_time: torch.Tensor,
    mpm_solver,
    mpm_state,
    gaussian_xyz,
    plane_mean,
    plane_normal,
    thres=0.2,
):
    """
    plane_mean and plane_normal are in original coordinate, not being normalized
    Args:
        xyzs_over_time: [T, N, 3]
        gaussian_xyz: [N, 3] torch.Tensor
        plane_mean: [3]
        plane_normal: [3]
        thres: float

    """

    plane_normal = plane_normal / torch.norm(plane_normal)
    # [n_particles]
    plane_dist = torch.abs(
        torch.sum(
            (gaussian_xyz - plane_mean.unsqueeze(0)) * plane_normal.unsqueeze(0), dim=-1
        )
    )
    # [n_particles]
    freeze_mask = plane_dist < thres
    freeze_mask = freeze_mask.type(torch.int)

    num_freeze_pts = freeze_mask.sum()
    print("num freeze pts from plannar boundary", num_freeze_pts.item())
    free_velocity = xyzs_over_time.new_zeros(3)
    # print("free velocity", free_velocity.shape, freeze_mask.shape)

    mpm_solver.enforce_particle_velocity_by_mask(
        mpm_state, freeze_mask, free_velocity, start_time=0, end_time=100000
    )

    return freeze_mask


def find_far_points(xyzs, selected_points, thres=0.05):
    """
    Args:
        xyzs: [N, 3]
        selected_points: [M, 3]
    Outs:
        freeze_mask: [N], 1 for points that are far away, 0 for points that are close
                    dtype=torch.int
    """
    chunk_size = 10000

    freeze_mask_list = []
    for i in range(0, xyzs.shape[0], chunk_size):

        end_index = min(i + chunk_size, xyzs.shape[0])
        xyzs_chunk = xyzs[i:end_index]
        # [M, N]
        cdist = torch.cdist(xyzs_chunk, selected_points)

        min_dist, _ = torch.min(cdist, dim=-1)
        freeze_mask = min_dist > thres
        freeze_mask = freeze_mask.type(torch.int)
        freeze_mask_list.append(freeze_mask)

    freeze_mask = torch.cat(freeze_mask_list, dim=0)

    # 1 for points that are far away, 0 for points that are close
    return freeze_mask


def setup_boundary_condition_with_points(
    xyzs, selected_points, mpm_solver, mpm_state, thres=0.05
):
    """
    Args:
        xyzs: [N, 3]
        selected_points: [M, 3]
    """

    freeze_mask = find_far_points(xyzs, selected_points, thres=thres)
    num_freeze_pts = freeze_mask.sum()
    print("num freeze pts from static points", num_freeze_pts.item())

    free_velocity = torch.zeros_like(xyzs[0])  # [3] in device

    mpm_solver.enforce_particle_velocity_by_mask(
        mpm_state, freeze_mask, free_velocity, start_time=0, end_time=1000000
    )

    return freeze_mask


def setup_bottom_boundary_condition(xyzs, mpm_solver, mpm_state, percentile=0.05):
    """
    Args:
        xyzs: [N, 3]
        selected_points: [M, 3]
    """
    max_z, min_z = torch.max(xyzs[:, 2]), torch.min(xyzs[:, 2])
    thres = min_z + (max_z - min_z) * percentile
    freeze_mask = xyzs[:, 2] < thres

    freeze_mask = freeze_mask.type(torch.int)
    num_freeze_pts = freeze_mask.sum()
    print("num freeze pts from bottom points", num_freeze_pts.item())

    free_velocity = torch.zeros_like(xyzs[0])  # [3] in device

    mpm_solver.enforce_particle_velocity_by_mask(
        mpm_state, freeze_mask, free_velocity, start_time=0, end_time=1000000
    )

    return freeze_mask


def render_single_view_video(
    cam,
    render_params,
    motion_model,
    time_stamps,
    rand_bg=False,
    render_flow=False,
    query_mask=None,
):
    """
    Args:
        cam:
        motion_model: Callable function, f(x, t) => translation, rotation
        time_stamps: [T]
        query_mask: Tensor of [N], 0 for freeze points, 1 for moving points
    Outs:
        ret_video: [T, 3, H, W] value in [0, 1]
    """

    if rand_bg:
        bg_color = torch.rand(3, device="cuda")
    else:
        bg_color = render_params.bg_color

    ret_img_list = []
    for time_stamp in time_stamps:
        if not render_flow:
            new_gaussians = render_params.gaussians.apply_se3_fields(
                motion_model, time_stamp
            )
            if query_mask is not None:
                new_gaussians._xyz = new_gaussians._xyz * query_mask.unsqueeze(
                    -1
                ) + render_params.gaussians._xyz * (1 - query_mask.unsqueeze(-1))
                new_gaussians._rotation = (
                    new_gaussians._rotation * query_mask.unsqueeze(-1)
                    + render_params.gaussians._rotation * (1 - query_mask.unsqueeze(-1))
                )
            # [3, H, W]
            img = render_gaussian(
                cam,
                new_gaussians,
                render_params.render_pipe,
                bg_color,
            )[
                "render"
            ]  # value in [0, 1]
        else:
            inp_time = (
                torch.ones_like(render_params.gaussians._xyz[:, 0:1]) * time_stamp
            )
            inp = torch.cat([render_params.gaussians._xyz, inp_time], dim=-1)
            # [bs, 3, 3]. [bs, 3]
            R, point_disp = motion_model(inp)

            img = render_flow_depth_w_gaussian(
                cam,
                render_params.gaussians,
                render_params.render_pipe,
                point_disp,
                bg_color,
            )["render"]

        ret_img_list.append(img[None, ...])

    ret_video = torch.cat(ret_img_list, dim=0)  # [T, 3, H, W]
    return ret_video


def render_gaussian_seq(cam, render_params, gaussian_pos_list, gaussian_cov_list):

    ret_img_list = []
    gaussians = render_params.gaussians
    old_xyz = gaussians._xyz.clone()
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    for i in range(len(gaussian_pos_list)):

        xyz = gaussian_pos_list[i]
        gaussians._xyz = xyz
        # TODO, how to deal with cov
        img = render_gaussian(
            cam,
            gaussians,
            render_params.render_pipe,
            background,
        )["render"]

        ret_img_list.append(img[None, ...])

    gaussians._xyz = old_xyz  # set back
    # [T, C, H, W], in [0, 1]
    rendered_video = torch.cat(ret_img_list, dim=0)

    return rendered_video


def render_gaussian_seq_w_mask(
    cam, render_params, gaussian_pos_list, gaussian_cov_list, update_mask
):

    ret_img_list = []
    gaussians = render_params.gaussians
    old_xyz = gaussians._xyz.clone()
    old_cov = gaussians.get_covariance().clone()

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    for i in range(len(gaussian_pos_list)):

        xyz = gaussian_pos_list[i]
        gaussians._xyz[update_mask, ...] = xyz

        if gaussian_cov_list is not None:
            cov = gaussian_cov_list[i]
            old_cov[update_mask, ...] = cov
            cov3D_precomp = old_cov

        else:
            cov3D_precomp = None

        img = render_gaussian(
            cam,
            gaussians,
            render_params.render_pipe,
            background,
            cov3D_precomp=cov3D_precomp,
        )["render"]

        ret_img_list.append(img[None, ...])

    gaussians._xyz = old_xyz  # set back
    # [T, C, H, W], in [0, 1]
    rendered_video = torch.cat(ret_img_list, dim=0)

    return rendered_video


def render_gaussian_seq_w_mask_with_disp(
    cam, render_params, orign_points, top_k_index, disp_list, update_mask
):
    """
    Args:
        cam: Camera or list of Camera
        orign_points: [m, 3]
        disp_list: List[m, 3]
        top_k_index: [n, top_k]

    """

    ret_img_list = []
    gaussians = render_params.gaussians
    old_xyz = gaussians._xyz.clone()
    old_rotation = gaussians._rotation.clone()

    query_pts = old_xyz[update_mask, ...]
    query_rotation = old_rotation[update_mask, ...]

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    for i in range(len(disp_list)):

        if isinstance(cam, list):
            render_cam = cam[i]
        else:
            render_cam = cam
        disp = disp_list[i]
        new_xyz, new_rotation = interpolate_points_w_R(
            query_pts, query_rotation, orign_points, disp, top_k_index
        )
        gaussians._xyz[update_mask, ...] = new_xyz
        gaussians._rotation[update_mask, ...] = new_rotation

        cov3D_precomp = None

        img = render_gaussian(
            render_cam,
            gaussians,
            render_params.render_pipe,
            background,
            cov3D_precomp=cov3D_precomp,
        )["render"]

        ret_img_list.append(img[None, ...])

    gaussians._xyz = old_xyz  # set back
    gaussians._rotation = old_rotation
    # [T, C, H, W], in [0, 1]
    rendered_video = torch.cat(ret_img_list, dim=0)

    return rendered_video


def render_gaussian_seq_w_mask_with_disp_for_figure(
    cam, render_params, orign_points, top_k_index, disp_list, update_mask
):
    """
    Args:
        cam: Camera or list of Camera
        orign_points: [m, 3]
        disp_list: List[m, 3]
        top_k_index: [n, top_k]

    """

    ret_img_list = []
    moving_img_list = []
    gaussians = render_params.gaussians
    old_xyz = gaussians._xyz.clone()
    old_rotation = gaussians._rotation.clone()

    query_pts = old_xyz[update_mask, ...]
    query_rotation = old_rotation[update_mask, ...]

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    background_black = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    for i in range(len(disp_list)):

        if isinstance(cam, list):
            render_cam = cam[i]
        else:
            render_cam = cam
        disp = disp_list[i]
        new_xyz, new_rotation = interpolate_points_w_R(
            query_pts, query_rotation, orign_points, disp, top_k_index
        )
        gaussians._xyz[update_mask, ...] = new_xyz
        gaussians._rotation[update_mask, ...] = new_rotation

        cov3D_precomp = None

        img = render_gaussian(
            render_cam,
            gaussians,
            render_params.render_pipe,
            background,
            cov3D_precomp=cov3D_precomp,
        )["render"]

        masked_gaussians = gaussians.apply_mask(update_mask)
        moving_img = render_gaussian(
            render_cam,
            masked_gaussians,
            render_params.render_pipe,
            background_black,
            cov3D_precomp=cov3D_precomp,
        )["render"]

        ret_img_list.append(img[None, ...])
        moving_img_list.append(moving_img[None, ...])

    gaussians._xyz = old_xyz  # set back
    gaussians._rotation = old_rotation
    # [T, C, H, W], in [0, 1]
    rendered_video = torch.cat(ret_img_list, dim=0)
    moving_part_video = torch.cat(moving_img_list, dim=0)

    return rendered_video, moving_part_video


def render_gaussian_seq_w_mask_cam_seq(
    cam_list, render_params, gaussian_pos_list, gaussian_cov_list, update_mask
):

    ret_img_list = []
    gaussians = render_params.gaussians
    old_xyz = gaussians._xyz.clone()
    old_cov = gaussians.get_covariance().clone()

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    for i in range(len(gaussian_pos_list)):

        xyz = gaussian_pos_list[i]
        gaussians._xyz[update_mask, ...] = xyz

        if gaussian_cov_list is not None:
            cov = gaussian_cov_list[i]
            old_cov[update_mask, ...] = cov
            cov3D_precomp = old_cov

        else:
            cov3D_precomp = None

        img = render_gaussian(
            cam_list[i],
            gaussians,
            render_params.render_pipe,
            background,
            cov3D_precomp=cov3D_precomp,
        )["render"]

        ret_img_list.append(img[None, ...])

    gaussians._xyz = old_xyz  # set back
    # [T, C, H, W], in [0, 1]
    rendered_video = torch.cat(ret_img_list, dim=0)

    return rendered_video


def apply_grid_bc_w_freeze_pts(grid_size, grid_lim, freeze_pts, mpm_solver):

    device = freeze_pts.device

    grid_pts_cnt = torch.zeros(
        (grid_size, grid_size, grid_size), dtype=torch.int32, device=device
    )

    dx = grid_lim / grid_size
    inv_dx = 1.0 / dx

    freeze_pts = (freeze_pts * inv_dx).long()

    for x, y, z in freeze_pts:
        grid_pts_cnt[x, y, z] += 1

    freeze_grid_mask = grid_pts_cnt >= 1

    freeze_grid_mask_int = freeze_grid_mask.type(torch.int32)

    number_freeze_grid = freeze_grid_mask_int.sum().item()
    print("number of freeze grid", number_freeze_grid)

    mpm_solver.enforce_grid_velocity_by_mask(freeze_grid_mask_int)

    # add debug section:

    return freeze_grid_mask


def add_constant_force(
    mpm_sovler,
    mpm_state,
    xyzs,
    center_point,
    radius,
    force,
    dt,
    start_time,
    end_time,
    device,
):
    """
    Args:
        xyzs: [N, 3]
        center_point: [3]
        radius: float
        force: [3]

    """

    # compute distance from xyzs to center_point
    # [N]
    dist = torch.norm(xyzs - center_point.unsqueeze(0), dim=-1)

    apply_force_mask = dist < radius
    apply_force_mask = apply_force_mask.type(torch.int)

    print(apply_force_mask.shape, apply_force_mask.sum().item(), "apply force mask")

    mpm_sovler.add_impulse_on_particles_with_mask(
        mpm_state,
        force,
        dt,
        apply_force_mask,
        start_time=start_time,
        end_time=end_time,
        device=device,
    )


@torch.no_grad()
def render_force_2d(cam, render_params, center_point, force):

    force_in_2d_scale = 80  # unit as pixel
    two_points = torch.stack([center_point, center_point + force], dim=0)

    gaussians = render_params.gaussians
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # [3, H, W]
    img = render_gaussian(
        cam,
        gaussians,
        render_params.render_pipe,
        background,
    )["render"]
    img = img.detach().contiguous()
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = img * 255
    img = img.astype(np.uint8).copy()

    # two_points.  [2, 3]
    # arrow_2d: [2, 2]
    arrow_2d = render_arrow_in_screen(cam, two_points)

    arrow_2d = arrow_2d.cpu().numpy()

    start, vec_2d = arrow_2d[0], arrow_2d[1] - arrow_2d[0]
    vec_2d = vec_2d / np.linalg.norm(vec_2d)

    start = start  # + np.array([540.0, 288.0])  # [W, H] / 2
    # debug here.
    # 1. unit in pixel?
    # 2. use cv2 to add arrow?
    # draw cirrcle at start in img

    # img = img.transpose(2, 0, 1)
    img = cv2.circle(img, (int(start[0]), int(start[1])), 40, (255, 255, 255), 8)

    # draw arrow in img
    end = start + vec_2d * force_in_2d_scale
    end = end.astype(np.int32)
    start = start.astype(np.int32)
    img = cv2.arrowedLine(img, (start[0], start[1]), (end[0], end[1]), (0, 255, 255), 8)

    return img


def render_gaussian_seq_w_mask_cam_seq_with_force(
    cam_list,
    render_params,
    gaussian_pos_list,
    gaussian_cov_list,
    update_mask,
    pts_index,
    force,
    force_steps,
):

    force_in_2d_scale = 80  # unit as pixel
    ret_img_list = []
    gaussians = render_params.gaussians
    old_xyz = gaussians._xyz.clone()
    old_cov = gaussians.get_covariance().clone()

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    for i in range(len(gaussian_pos_list)):

        xyz = gaussian_pos_list[i]
        gaussians._xyz[update_mask, ...] = xyz

        if gaussian_cov_list is not None:
            cov = gaussian_cov_list[i]
            old_cov[update_mask, ...] = cov
            cov3D_precomp = old_cov

        else:
            cov3D_precomp = None

        img = render_gaussian(
            cam_list[i],
            gaussians,
            render_params.render_pipe,
            background,
            cov3D_precomp=cov3D_precomp,
        )["render"]

        # to [H, W, 3]
        img = img.detach().contiguous().cpu().numpy().transpose(1, 2, 0)
        img = np.clip((img * 255), 0, 255).astype(np.uint8).copy()

        if i < force_steps:
            center_point = gaussians._xyz[pts_index]
            two_points = torch.stack([center_point, center_point + force], dim=0)

            arrow_2d = render_arrow_in_screen(cam_list[i], two_points)

            arrow_2d = arrow_2d.cpu().numpy()

            start, vec_2d = arrow_2d[0], arrow_2d[1] - arrow_2d[0]
            vec_2d = vec_2d / np.linalg.norm(vec_2d)

            start = start  # + np.array([540.0, 288.0])

            img = cv2.circle(
                img, (int(start[0]), int(start[1])), 40, (255, 255, 255), 8
            )

            # draw arrow in img
            end = start + vec_2d * force_in_2d_scale
            end = end.astype(np.int32)
            start = start.astype(np.int32)
            img = cv2.arrowedLine(
                img, (start[0], start[1]), (end[0], end[1]), (0, 255, 255), 8
            )

        img = img.transpose(2, 0, 1)
        ret_img_list.append(img[None, ...])

    gaussians._xyz = old_xyz  # set back
    # [T, C, H, W], in [0, 1]
    rendered_video = np.concatenate(ret_img_list, axis=0)

    return rendered_video


def render_gaussian_seq_w_mask_cam_seq_with_force_with_disp(
    cam_list,
    render_params,
    orign_points,
    top_k_index,
    disp_list,
    update_mask,
    pts_index,
    force,
    force_steps,
):

    force_in_2d_scale = 80  # unit as pixel
    ret_img_list = []
    gaussians = render_params.gaussians
    old_xyz = gaussians._xyz.clone()
    old_rotation = gaussians._rotation.clone()

    query_pts = old_xyz[update_mask, ...]
    query_rotation = old_rotation[update_mask, ...]

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    for i in range(len(disp_list)):

        disp = disp_list[i]
        new_xyz, new_rotation = interpolate_points_w_R(
            query_pts, query_rotation, orign_points, disp, top_k_index
        )
        gaussians._xyz[update_mask, ...] = new_xyz
        gaussians._rotation[update_mask, ...] = new_rotation

        cov3D_precomp = None

        img = render_gaussian(
            cam_list[i],
            gaussians,
            render_params.render_pipe,
            background,
            cov3D_precomp=cov3D_precomp,
        )["render"]

        # to [H, W, 3]
        img = img.detach().contiguous().cpu().numpy().transpose(1, 2, 0)
        img = np.clip((img * 255), 0, 255).astype(np.uint8).copy()

        if i < force_steps:
            center_point = gaussians._xyz[pts_index]
            two_points = torch.stack([center_point, center_point + force], dim=0)

            arrow_2d = render_arrow_in_screen(cam_list[i], two_points)

            arrow_2d = arrow_2d.cpu().numpy()

            start, vec_2d = arrow_2d[0], arrow_2d[1] - arrow_2d[0]
            vec_2d = vec_2d / np.linalg.norm(vec_2d)

            start = start  # + np.array([540.0, 288.0])

            img = cv2.circle(
                img, (int(start[0]), int(start[1])), 40, (255, 255, 255), 5
            )

            # draw arrow in img
            end = start + vec_2d * force_in_2d_scale
            end = end.astype(np.int32)
            start = start.astype(np.int32)
            img = cv2.arrowedLine(
                img, (start[0], start[1]), (end[0], end[1]), (255, 255, 0), 4
            )

        img = img.transpose(2, 0, 1)
        ret_img_list.append(img[None, ...])

    gaussians._xyz = old_xyz  # set back
    gaussians._rotation = old_rotation
    # [T, C, H, W], in [0, 1]
    rendered_video = np.concatenate(ret_img_list, axis=0)

    return rendered_video


def downsample_with_kmeans(points_array: np.ndarray, num_points: int):
    """
    Args:
        points_array: [N, 3]
        num_points: int
    Outs:
        downsampled_points: [num_points, 3]
    """

    print(
        "=> staring downsample with kmeans from ",
        points_array.shape[0],
        " points to ",
        num_points,
        " points",
    )
    s_time = time()
    kmeans = KMeans(n_clusters=num_points, random_state=0).fit(points_array)
    cluster_centers = kmeans.cluster_centers_
    e_time = time()

    print("=> downsample with kmeans takes ", e_time - s_time, " seconds")
    return cluster_centers


@torch.no_grad()
def downsample_with_kmeans_gpu(points_array: torch.Tensor, num_points: int):

    from kmeans_gpu import KMeans

    kmeans = KMeans(
        n_clusters=num_points,
        max_iter=100,
        tolerance=1e-4,
        distance="euclidean",
        sub_sampling=None,
        max_neighbors=15,
    )

    features = torch.ones(1, 1, points_array.shape[0], device=points_array.device)
    points_array = points_array.unsqueeze(0)
    # Forward

    print(
        "=> staring downsample with kmeans from ",
        points_array.shape[1],
        " points to ",
        num_points,
        " points",
    )
    s_time = time()
    centroids, features = kmeans(points_array, features)

    ret_points = centroids.squeeze(0)
    e_time = time()
    print("=> downsample with kmeans takes ", e_time - s_time, " seconds")

    # [np_subsample, 3]
    return ret_points


def interpolate_points(query_points, drive_displacement, top_k_index):
    """
    Args:
        query_points: [n, 3]
        drive_displacement: [m, 3]
        top_k_index: [n, top_k] < m
    """

    top_k_disp = drive_displacement[top_k_index]

    t = top_k_disp.mean(dim=1)

    ret_points = query_points + t

    return ret_points


def interpolate_points_w_R(
    query_points, query_rotation, drive_origin_pts, drive_displacement, top_k_index
):
    """
    Args:
        query_points: [n, 3]
        drive_origin_pts: [m, 3]
        drive_displacement: [m, 3]
        top_k_index: [n, top_k] < m

    Or directly call: apply_discrete_offset_filds_with_R(self, origin_points, offsets, topk=6):
        Args:
            origin_points: (N_r, 3)
            offsets: (N_r, 3)
        in rendering
    """

    # [n, topk, 3]
    top_k_disp = drive_displacement[top_k_index]
    source_points = drive_origin_pts[top_k_index]

    R, t = get_rigid_transform(source_points, source_points + top_k_disp)

    avg_offsets = top_k_disp.mean(dim=1)

    ret_points = query_points + avg_offsets

    new_rotation = quaternion_multiply(matrix_to_quaternion(R), query_rotation)

    return ret_points, new_rotation


def create_camera_path(
    cam,
    radius: float,
    focus_pt: np.ndarray = np.array([0, 0, 0]),
    up: np.ndarray = np.array([0, 0, 1]),
    n_frames: int = 60,
    n_rots: int = 1,
    y_scale: float = 1.0,
):

    R, T = cam.R, cam.T
    # R, T = R.cpu().numpy(), T.cpu().numpy()

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    C2W[:3, 1:3] *= -1

    import copy
    from motionrep.utils.camera_utils import generate_spiral_path
    from motionrep.data.cameras import Camera

    lookat_pt = focus_pt
    render_poses = generate_spiral_path(
        C2W, radius, lookat_pt, up, n_frames, n_rots, y_scale
    )

    FoVy, FoVx = cam.FoVy, cam.FoVx
    height, width = cam.image_height, cam.image_width

    ret_cam_list = []
    for i in range(n_frames):
        c2w_opengl = render_poses[i]
        c2w = copy.deepcopy(c2w_opengl)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        cam = Camera(
            R=R,
            T=T,
            FoVy=FoVy,
            FoVx=FoVx,
            img_path=None,
            img_hw=(height, width),
            timestamp=None,
            data_device="cuda",
        )
        ret_cam_list.append(cam)

    return ret_cam_list


def get_camera_trajectory(cam, num_pos, camera_cfg: dict, dataset):
    if camera_cfg["type"] == "spiral":
        interpolated_cameras = create_camera_path(
            cam,
            radius=camera_cfg["radius"],
            focus_pt=camera_cfg["focus_point"],
            up=camera_cfg["up"],
            n_frames=num_pos,
        )
    elif camera_cfg["type"] == "interpolation":
        if "start_frame" in camera_cfg and "end_frame" in camera_cfg:
            interpolated_cameras = dataset.interpolate_camera(
                camera_cfg["start_frame"], camera_cfg["end_frame"], num_pos
            )
        else:
            interpolated_cameras = dataset.interpolate_camera(
                camera_cfg["start_frame"], camera_cfg["start_frame"], num_pos
            )

    print(
        "number of simulated frames: ",
        num_pos,
        "num camera viewpoints: ",
        len(interpolated_cameras),
    )
    return interpolated_cameras
