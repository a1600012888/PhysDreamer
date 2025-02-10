import time
import numpy as np
from fire import Fire
import os
import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
import torch
from tqdm import tqdm


def load_gaussians(input_dir: str = None):
    name_dict = {
        "position": "pos.npy",  # [T, N, 3]
        "rotation": "rot.npy",  # [T, N, 4]
        "cov": "cov.npy",  # [T, N, 6]
    }

    assert os.path.exists(input_dir), "Input directory does not exist"

    ret_dict = {}

    for key, value in name_dict.items():
        ret_dict[key] = np.load(os.path.join(input_dir, value))

    pos_max = ret_dict["position"].max()
    pos_min = ret_dict["position"].min()

    # ret_dict["position"] = (ret_dict["position"]) / (pos_max - pos_min) * 0.5 - pos_min / (pos_max - pos_min) * 0.5 + 0.1
    # ret_dict["cov"] = ret_dict["cov"] / (pos_max - pos_min) * 0.5
    scale = (pos_max - pos_min) * 2.0
    shift = -pos_min

    ret_dict["position"] = (ret_dict["position"] + shift) / scale
    ret_dict["cov"] = ret_dict["cov"] / scale

    # pos_new = (pos + shift ) / scale
    # pos_orign = pos_new * scale  - shift
    return ret_dict, scale, shift


def init_volume(xyz, grid=[-1, 1], num_grid=20):
    pass


def run_mpm_gaussian(input_dir, output_dir=None, fps=6, device=0):
    wp.init()
    wp.config.verify_cuda = True

    device = "cuda:{}".format(device)

    gaussian_dict, scale, shift = load_gaussians(input_dir)

    velocity_scaling = 10
    velocity = (
        (gaussian_dict["position"][1:] - gaussian_dict["position"][:-1])
        / fps
        * velocity_scaling
    )

    velocity_abs = np.abs(velocity)
    print(
        "velocity mean-max-min",
        velocity_abs.mean(),
        velocity_abs.max(),
        velocity_abs.min(),
    )

    init_velocity = velocity[0]
    init_position = gaussian_dict["position"][0]
    init_rotation = gaussian_dict["rotation"][0]
    init_cov = gaussian_dict["cov"][0]
    tensor_init_pos = torch.from_numpy(init_position).float().to(device)
    tensor_init_cov = torch.from_numpy(init_cov).float().to(device)
    tensor_init_velocity = torch.from_numpy(init_velocity).float().to(device)

    # print(tensor_init_pos.max(), tensor_init_pos.min(), tensor_init_pos.shape)

    mpm_solver = MPM_Simulator_WARP(
        10
    )  # initialize with whatever number is fine. it will be reintialized

    # TODO, Compute volume later
    volume_tensor = (
        torch.ones(
            init_velocity.shape[0],
        )
        * 2.5e-8  # m^3
    )

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

    position_tensor = mpm_solver.export_particle_x_to_torch()
    velo = wp.to_torch(mpm_solver.mpm_state.particle_v)
    cov = wp.to_torch(mpm_solver.mpm_state.particle_init_cov)
    print(
        "pos in box: ",
        position_tensor.max(),
        position_tensor.min(),
    )

    material_params = {
        "E": 0.0002,  # 0.1-200 MPa
        "nu": 0.4,  # > 0.35
        "material": "jelly",
        # "friction_angle": 25,
        "g": [0.0, 0.0, 0],
        "density": 1,  # kg / m^3
    }

    print("pre set")
    mpm_solver.set_parameters_dict(material_params)
    print("set")
    mpm_solver.finalize_mu_lam()  # set mu and lambda from the E and nu input
    print("finalize")
    # mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0,0.0,1.0), 'sticky', 0.0)

    if output_dir is None:
        output_dir = "./gaussian_sim_results"
    os.makedirs(output_dir, exist_ok=True)

    # save_data_at_frame(mpm_solver, output_dir, 0, save_to_ply=True, save_to_h5=False)
    pos_list = []
    pos = mpm_solver.export_particle_x_to_torch().clone()
    pos = (pos * scale) - shift
    pos_list.append(pos.detach().clone())

    total_time = 20
    time_step = 0.002
    total_iters = int(total_time / time_step)

    for k in tqdm(range(1, total_iters)):
        mpm_solver.p2g2p(k, time_step, device=device)

        if k % 50 == 0:
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
    dataset_dir="../../data/physics_dreamer/llff_flower_undistorted",
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
        save_path = "test.mp4"
    else:
        save_path = save_name + ".mp4"
    print("save video to ", save_path)
    save_video_imageio(save_path, video_numpy, fps=10)


if __name__ == "__main__":
    Fire(run_mpm_gaussian)
    # Fire(code_test)
