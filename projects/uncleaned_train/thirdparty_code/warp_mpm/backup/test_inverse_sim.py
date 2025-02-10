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
import pickle


def test(
    input_dir,
    pickle_path="output/E_0.2_nu_0.1_material_jelly_density_0.2__timestep_0.001_vs0.5_totaltime_10.pkl",
    output_dir=None,
    fps=6,
    device=0,
):
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
        "E": 2.5,  # 0.1-200 MPa
        "nu": 0.2,  # > 0.35
        "material": "jelly",
        # "material": "metal",
        # "friction_angle": 25,
        "g": [0.0, 0.0, 0],
        "density": 0.2,  # kg / m^3
    }

    n_particles = tensor_init_pos.shape[0]
    mpm_state = MPMStateStruct()

    mpm_state.init(init_position.shape[0], device=device, requires_grad=True)
    mpm_state.from_torch(
        tensor_init_pos.clone(),
        volume_tensor,
        tensor_init_cov,
        tensor_init_velocity.clone(),
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

    total_time = 0.02
    time_step = 0.001
    total_iters = int(total_time / time_step)
    total_iters = 10

    dt = time_step
    with open(pickle_path, "rb") as f:
        gt_dict = pickle.load(f)

    sim_sub_step = 10
    gt_pos_numpy_list = gt_dict["pos_list"]
    pos_gt_1 = gt_pos_numpy_list[sim_sub_step - 1]
    pos_gt_1_warp = wp.from_numpy(
        pos_gt_1, dtype=wp.vec3, device=device, requires_grad=True
    )

    E_cur = material_params["E"]
    nu_cur = material_params["nu"]

    init_lr = 3e-6
    total_train_steps = 2000
    for train_step in range(total_train_steps):
        learning_rate = (
            (total_train_steps - train_step + 1) / total_train_steps * init_lr
        )
        tape = MyTape()  # wp.Tape()
        with tape:
            # for k in tqdm(range(1, total_iters)):
            k = 0
            mpm_solver.time = 0.0

            mpm_solver.set_E_nu(mpm_model, E_cur, nu_cur, device=device)
            for k in range(sim_sub_step):
                mpm_solver.p2g2p(mpm_model, mpm_state, k, time_step, device=device)

            loss = torch.zeros(1, device=device)
            loss = wp.from_torch(loss, requires_grad=True)
            wp.launch(
                position_loss_kernel,
                dim=n_particles,
                inputs=[mpm_state, pos_gt_1_warp, loss],
                device=device,
            )

        tape.backward(loss)  # 75120.86

        E_grad = wp.from_torch(torch.zeros(1, device=device), requires_grad=False)
        nu_grad = wp.from_torch(torch.zeros(1, device=device), requires_grad=False)

        wp.launch(
            aggregate_grad,
            dim=n_particles,
            inputs=[
                E_grad,
                mpm_model.E.grad,
            ],
            device=device,
        )
        wp.launch(
            aggregate_grad,
            dim=n_particles,
            inputs=[nu_grad, mpm_model.nu.grad],
            device=device,
        )

        E_cur = E_cur - wp.to_torch(E_grad).item() * learning_rate
        nu_cur = nu_cur - wp.to_torch(nu_grad).item() * learning_rate
        # clip:
        E_cur = max(1e-5, min(E_cur, 200))
        nu_cur = max(1e-2, min(nu_cur, 0.449))

        tape.zero()
        print(
            loss,
            "pre backward",
            E_cur,
            nu_cur,
            E_grad,
            nu_grad,
        )

        mpm_state.reset_state(
            tensor_init_pos.clone(),
            tensor_init_cov,
            tensor_init_velocity.clone(),
            device=device,
            requires_grad=True,
        )
        # might need to set mpm_model.yield_stress

    from IPython import embed  #     embed()


@wp.kernel
def position_loss_kernel(
    mpm_state: MPMStateStruct,
    gt_pos: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    pos = mpm_state.particle_x[tid]
    pos_gt = gt_pos[tid]

    # l1_diff = wp.abs(pos - pos_gt)
    l2 = wp.length(pos - pos_gt)

    wp.atomic_add(loss, 0, l2)


@wp.kernel
def step_kernel(x: wp.array(dtype=float), grad: wp.array(dtype=float), alpha: float):
    tid = wp.tid()

    # gradient descent step
    x[tid] = x[tid] - grad[tid] * alpha


@wp.kernel
def aggregate_grad(x: wp.array(dtype=float), grad: wp.array(dtype=float)):
    tid = wp.tid()

    # gradient descent step
    wp.atomic_add(x, 0, grad[tid])


if __name__ == "__main__":
    Fire(test)
