import warp as wp
import numpy as np
import torch
import os
from mpm_solver_warp_diff import MPM_Simulator_WARPDiff
from run_gaussian_static import load_gaussians, get_volume
from tqdm import tqdm
from fire import Fire

from diff_warp_utils import MPMStateStruct, MPMModelStruct
from warp_rewrite import MyTape

from mpm_utils import *
import random


def test(input_dir, output_dir=None, fps=6, device=0):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
        volume_array = np.load(volume_array_path)
        volume_tensor = torch.from_numpy(volume_array).float().to(device)
    else:
        volume_array = get_volume(init_position)
        np.save(volume_array_path, volume_array)
        volume_tensor = torch.from_numpy(volume_array).float().to(device)

    tensor_init_pos = torch.from_numpy(init_position).float().to(device)
    tensor_init_cov = torch.from_numpy(init_cov).float().to(device)
    tensor_init_velocity = torch.from_numpy(init_velocity).float().to(device)

    material_params = {
        "E": 2.0,  # 0.1-200 MPa
        "nu": 0.1,  # > 0.35
        "material": "jelly",
        # "material": "metal",
        # "friction_angle": 25,
        "g": [0.0, 0.0, 0],
        "density": 0.02,  # kg / m^3
    }

    n_particles = tensor_init_pos.shape[0]
    mpm_state = MPMStateStruct()

    mpm_state.init(init_position.shape[0], device=device, requires_grad=True)
    mpm_state.from_torch(
        tensor_init_pos,
        volume_tensor,
        tensor_init_cov,
        tensor_init_velocity,
        device=device,
        requires_grad=True,
        n_grid=100,
        grid_lim=1.0,
    )

    mpm_model = MPMModelStruct()
    mpm_model.init(n_particles, device=device, requires_grad=True)
    mpm_model.init_other_params(n_grid=100, grid_lim=1.0, device=device)

    E_tensor = (torch.ones(n_particles) * material_params["E"]).contiguous().to(device)
    nu_tensor = (
        (torch.ones(n_particles) * material_params["nu"]).contiguous().to(device)
    )
    mpm_model.from_torch(E_tensor, nu_tensor, device=device, requires_grad=True)

    mpm_solver = MPM_Simulator_WARPDiff(
        n_particles, n_grid=100, grid_lim=1.0, device=device
    )

    mpm_solver.set_parameters_dict(mpm_model, mpm_state, material_params)

    # set boundary conditions
    static_center_point = (
        torch.from_numpy(gaussian_dict["satic_center_point"]).float().to(device)
    )
    max_static_offset = (
        torch.from_numpy(gaussian_dict["max_static_offset"]).float().to(device)
    )
    velocity = torch.zeros_like(static_center_point)
    mpm_solver.enforce_particle_velocity_translation(
        mpm_state,
        static_center_point,
        max_static_offset,
        velocity,
        start_time=0,
        end_time=1000,
        device=device,
    )

    mpm_state.set_require_grad(True)

    total_time = 0.1
    time_step = 0.001
    total_iters = int(total_time / time_step)
    total_iters = 3
    loss = torch.zeros(1, device=device)
    loss = wp.from_torch(loss, requires_grad=True)

    dt = time_step
    tape = MyTape()  # wp.Tape()

    with tape:
        # for k in tqdm(range(1, total_iters)):
        k = 1
        # mpm_solver.p2g2p(k, time_step, device=device)
        for k in range(10):
            mpm_solver.p2g2p(mpm_model, mpm_state, k, time_step, device=device)

        wp.launch(
            position_loss_kernel,
            dim=n_particles,
            inputs=[mpm_state, loss],
            device=device,
        )

    print(loss, "pre backward")

    tape.backward(loss)  # 75120.86

    print(loss)

    v_grad = mpm_state.particle_v.grad
    x_grad = mpm_state.particle_x.grad
    e_grad = mpm_model.E.grad
    e_grad_torch = wp.to_torch(e_grad)
    grid_v_grad = mpm_state.grid_v_out.grad
    grid_v_in_grad = mpm_state.grid_v_in.grad
    print(x_grad)
    from IPython import embed

    embed()


@wp.kernel
def position_loss_kernel(mpm_state: MPMStateStruct, loss: wp.array(dtype=float)):
    tid = wp.tid()

    pos = mpm_state.particle_x[tid]
    wp.atomic_add(loss, 0, pos[0] + pos[1] + pos[2])
    # wp.atomic_add(loss, 0, mpm_state.particle_x[tid][0])


if __name__ == "__main__":
    Fire(test)
