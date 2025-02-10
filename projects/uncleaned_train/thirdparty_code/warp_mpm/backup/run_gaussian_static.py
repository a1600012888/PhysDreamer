import time
import numpy as np
from fire import Fire
import os
import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
import torch
from tqdm import tqdm
from motionrep.gaussian_3d.scene import GaussianModel


def load_gaussians(input_dir: str = None):
    if not input_dir.endswith("ply"):
        gaussian_path = os.path.join(input_dir, "point_cloud.ply")
    else:
        gaussian_path = input_dir

    gaussians = GaussianModel(3)

    gaussians.load_ply(gaussian_path)
    gaussians.detach_grad()

    pos = gaussians._xyz.detach().cpu().numpy()

    pos_max = pos.max()
    pos_min = pos.min()

    scale = (pos_max - pos_min) * 2.5
    shift = -pos_min + (pos_max - pos_min) * 0.25

    pos = (pos + shift) / scale

    cov = gaussians.get_covariance().detach().cpu().numpy()
    cov = cov / scale

    velocity = np.zeros_like(pos)

    height = pos[:, 2] - pos[:, 2].min()
    height_thres = 10
    velocity_mask = height > np.percentile(height, height_thres)

    static_points = pos[np.logical_not(velocity_mask)]

    static_points_mean = static_points.mean(axis=0)
    static_points_dist = static_points - static_points_mean
    max_static_offset = np.abs(static_points_dist).max(axis=0) * 0.8

    # boundary condition, set velocity to 0

    # x_velocity = np.sqrt(np.abs(pos[:, 0]) + 1e-8) * np.sign(pos[:, 0])
    # x_velocity = np.sqrt(height) * 0.1
    x_velocity = height**0.2 * 0.1
    velocity[velocity_mask, 0] = x_velocity[velocity_mask]
    velocity[velocity_mask, 1] = x_velocity[velocity_mask]

    ret_dict = {
        "position": pos,  # numpy [N, 3]
        "cov": cov,  # numpy [N, 6]
        "velocity": velocity,  # numpy [N, 3]
        "satic_center_point": static_points_mean,  # numpy [3]
        "max_static_offset": max_static_offset,  # numpy [3]
    }

    return ret_dict, scale, shift


def get_volume(xyzs: np.ndarray, resolution=128) -> np.ndarray:
    print("Compute Volume for each points")
    voxel_counts = np.zeros((resolution, resolution, resolution))

    points_xyzindex = ((xyzs + 1) / 2 * (resolution - 1)).astype(np.uint32)

    for x, y, z in points_xyzindex:
        voxel_counts[x, y, z] += 1

    points_number_in_corresponding_voxel = voxel_counts[
        points_xyzindex[:, 0], points_xyzindex[:, 1], points_xyzindex[:, 2]
    ]

    cell_volume = (2.0 / (resolution - 1)) ** 3

    points_volume = cell_volume / points_number_in_corresponding_voxel

    points_volume = points_volume.astype(np.float32)

    print(
        "mean volume",
        points_volume.mean(),
        "max volume",
        points_volume.max(),
        "min volume",
        points_volume.min(),
    )

    return points_volume


def run_mpm_gaussian(input_dir, output_dir=None, fps=6, device=0):
    wp.init()
    wp.config.verify_cuda = True

    device = "cuda:{}".format(device)

    gaussian_dict, scale, shift = load_gaussians(input_dir)

    velocity_scaling = 0.5

    init_velocity = velocity_scaling * gaussian_dict["velocity"]
    init_position = gaussian_dict["position"]
    init_cov = gaussian_dict["cov"]

    volume_array_path = os.path.join(input_dir, "volume_array.npy")
    if os.path.exists(volume_array_path):
        volume_tensor = torch.from_numpy(np.load(volume_array_path)).float().to(device)
    else:
        volume_array = get_volume(init_position)
        np.save(volume_array_path, volume_array)
        volume_tensor = torch.from_numpy(volume_array).float().to(device)

    tensor_init_pos = torch.from_numpy(init_position).float().to(device)
    tensor_init_cov = torch.from_numpy(init_cov).float().to(device)
    tensor_init_velocity = torch.from_numpy(init_velocity).float().to(device)

    print(
        "init position:",
        tensor_init_pos.max(),
        tensor_init_pos.min(),
        tensor_init_pos.shape,
    )
    velocity_abs = np.abs(init_velocity)
    print(
        "velocity mean-max-min",
        velocity_abs.mean(),
        velocity_abs.max(),
        velocity_abs.min(),
    )

    mpm_solver = MPM_Simulator_WARP(
        10
    )  # initialize with whatever number is fine. it will be reintialized

    mpm_solver.load_initial_data_from_torch(
        tensor_init_pos,
        volume_tensor,
        tensor_init_cov,
        tensor_init_velocity,
        device=device,
    )
    # mpm_solver.load_initial_data_from_torch(
    #     tensor_init_pos, volume_tensor, device=device
    # )

    # set boundary conditions
    static_center_point = (
        torch.from_numpy(gaussian_dict["satic_center_point"]).float().to(device)
    )
    max_static_offset = (
        torch.from_numpy(gaussian_dict["max_static_offset"]).float().to(device)
    )
    velocity = torch.zeros_like(static_center_point)
    mpm_solver.enforce_particle_velocity_translation(
        static_center_point,
        max_static_offset,
        velocity,
        start_time=0,
        end_time=1000,
        device=device,
    )

    position_tensor = mpm_solver.export_particle_x_to_torch()
    velo = wp.to_torch(mpm_solver.mpm_state.particle_v)
    cov = wp.to_torch(mpm_solver.mpm_state.particle_init_cov)
    print(
        "pos in box: ",
        position_tensor.max(),
        position_tensor.min(),
    )

    material_params = {
        "E": 0.2,  # 0.1-200 MPa
        "nu": 0.1,  # > 0.35
        "material": "jelly",
        # "material": "metal",
        # "friction_angle": 25,
        "g": [0.0, 0.0, 0],
        "density": 0.2,  # kg / m^3
    }

    print("pre set")
    mpm_solver.set_parameters_dict(material_params)
    print("set")
    mpm_solver.finalize_mu_lam()  # set mu and lambda from the E and nu input
    print("finalize")
    # mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0,0.0,1.0), 'sticky', 0.0)

    if output_dir is None:
        output_dir = "../../output/gaussian_sim_results"
    os.makedirs(output_dir, exist_ok=True)

    # save_data_at_frame(mpm_solver, output_dir, 0, save_to_ply=True, save_to_h5=False)
    pos_list = []
    pos = mpm_solver.export_particle_x_to_torch().clone()
    pos = (pos * scale) - shift
    pos_list.append(pos.detach().clone())

    total_time = 10
    time_step = 0.001
    total_iters = int(total_time / time_step)

    save_dict = {
        "pos_init": mpm_solver.export_particle_x_to_torch()
        .clone()
        .detach()
        .cpu()
        .numpy(),
        "velo_init": mpm_solver.export_particle_v_to_torch()
        .clone()
        .detach()
        .cpu()
        .numpy(),
        "pos_list": [],
    }

    for k in tqdm(range(1, total_iters)):
        mpm_solver.p2g2p(k, time_step, device=device)

        if k < 20:
            pos = mpm_solver.export_particle_x_to_torch().clone().detach().cpu().numpy()
            save_dict["pos_list"].append(pos)

        if k % 100 == 0:
            pos = mpm_solver.export_particle_x_to_torch().clone()
            pos = (pos * scale) - shift
            pos_list.append(pos.detach().clone())
            print(k)
            print(pos.max().item(), pos.min().item(), pos.mean().item())
        # save_data_at_frame(mpm_solver, output_dir, k, save_to_ply=True, save_to_h5=False)

    save_name = ""
    for key, value in material_params.items():
        if key == "g":
            continue
        save_name += "{}_{}_".format(key, value)

    save_name += "_timestep_{}_vs{}_totaltime_{}".format(
        time_step, velocity_scaling, total_time
    )

    render_gaussians(pos_list, save_name)

    # save sim data:
    save_path = os.path.join(output_dir, save_name + ".pkl")
    import pickle

    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f)


def code_test(input_dir, device=0):
    device = "cuda:{}".format(device)
    gaussian_dict, scale, shift = load_gaussians(input_dir)
    pos = gaussian_dict["position"]

    pos = (pos * scale) - shift

    pos = torch.from_numpy(pos).float().to(device)

    render_gaussians(pos)


def render_gaussians(
    pos_list,
    save_name=None,
    # dataset_dir="../../data/physics_dreamer/llff_flower_undistorted",
    dataset_dir="../../data/physics_dreamer/ficus",
):
    from motionrep.data.datasets.multiview_dataset import MultiviewImageDataset
    from motionrep.data.datasets.multiview_dataset import (
        camera_dataset_collate_fn as camera_dataset_collate_fn_img,
    )

    from motionrep.gaussian_3d.gaussian_renderer.render import render_gaussian
    from motionrep.gaussian_3d.scene import GaussianModel
    from typing import NamedTuple

    gaussian_path = os.path.join(dataset_dir, "point_cloud.ply")
    test_dataset = MultiviewImageDataset(
        dataset_dir,
        use_white_background=False,
        resolution=[576, 1024],
        use_index=list(range(5, 30, 4)),
        scale_x_angle=1.5,
    )
    print(
        "len of train dataset",
        len(test_dataset),
        "len of test dataset",
        len(test_dataset),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        collate_fn=camera_dataset_collate_fn_img,
    )

    class RenderPipe(NamedTuple):
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False

    class RenderParams(NamedTuple):
        render_pipe: RenderPipe
        bg_color: bool
        gaussians: GaussianModel
        camera_list: list

    gaussians = GaussianModel(3)
    camera_list = test_dataset.camera_list

    gaussians.load_ply(gaussian_path)
    gaussians.detach_grad()
    print(
        "load gaussians from: {}".format(gaussian_path),
        "... num gaussians: ",
        gaussians._xyz.shape[0],
    )
    bg_color = [1, 1, 1] if False else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    render_pipe = RenderPipe()

    render_params = RenderParams(
        render_pipe=render_pipe,
        bg_color=background,
        gaussians=gaussians,
        camera_list=camera_list,
    )

    data = next(iter(test_dataloader))
    cam = data["cam"][0]

    ret_img_list = []

    for i in range(len(pos_list) + 1):
        if i > 0:
            xyz = pos_list[i - 1]
            gaussians._xyz = xyz

        img = render_gaussian(
            cam,
            gaussians,
            render_params.render_pipe,
            background,
        )["render"]

        ret_img_list.append(img)

    # [T, C, H, W]
    video_array = torch.stack(ret_img_list, dim=0)
    video_numpy = video_array.detach().cpu().numpy() * 255
    video_numpy = np.clip(video_numpy, 0, 255).astype(np.uint8)

    video_numpy = np.transpose(video_numpy, [0, 2, 3, 1])
    from motionrep.utils.io_utils import save_video_imageio

    if save_name is None:
        save_path = "output/test.mp4"
    else:
        save_path = os.path.join("output", save_name + ".mp4")
    print("save video to ", save_path)
    save_video_imageio(save_path, video_numpy, fps=10)


if __name__ == "__main__":
    Fire(run_mpm_gaussian)
    # Fire(code_test)
