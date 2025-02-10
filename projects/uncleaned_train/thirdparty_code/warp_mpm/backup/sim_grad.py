import warp as wp
import numpy as np
import torch
import os
from mpm_solver_warp_diff import MPM_Simulator_WARPDiff
from run_gaussian_static import load_gaussians, get_volume
from tqdm import tqdm 
from fire import Fire 
from mpm_utils import *

from typing import Optional

import warp as wp


class MyTape(wp.Tape):

    # returns the adjoint of a kernel parameter
    def get_adjoint(self, a):
        if not wp.types.is_array(a) and not isinstance(a, wp.codegen.StructInstance):
            # if input is a simple type (e.g.: float, vec3, etc) then
            # no gradient needed (we only return gradients through arrays and structs)
            print("input is a simple type", type(a))
            return a

        elif wp.types.is_array(a) and a.grad:
            # keep track of all gradients used by the tape (for zeroing)
            # ignore the scalar loss since we don't want to clear its grad
            self.gradients[a] = a.grad
            return a.grad

        elif isinstance(a, wp.codegen.StructInstance):
            adj = a._cls()
            for name, _ in a._cls.ctype._fields_:
                if name.startswith("_"):
                    continue
                if isinstance(a._cls.vars[name].type, wp.array):
                    arr = getattr(a, name)
                    if arr is None:
                        continue
                    if arr.grad:
                        grad = self.gradients[arr] = arr.grad
                    else:
                        grad = wp.zeros_like(arr)
                    setattr(adj, name, grad)
                else:
                    setattr(adj, name, getattr(a, name))

            self.gradients[a] = adj
            return adj

        return None



def test(input_dir, output_dir=None, fps=6, device=0):
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

    mpm_solver = MPM_Simulator_WARPDiff(10)  # initialize with whatever number is fine. it will be reintialized

    tensor_init_pos.requires_grad = True
    tensor_init_cov.requires_grad = False
    tensor_init_velocity.requires_grad = True
    mpm_solver.load_initial_data_from_torch(
        tensor_init_pos,
        volume_tensor,
        tensor_init_cov,
        tensor_init_velocity,
        device=device,
    )
    mpm_solver.mpm_state.particle_x = wp.from_numpy(init_position, dtype=wp.vec3, requires_grad=True, device=device)
    mpm_solver.mpm_state.particle_v = wp.from_numpy(init_velocity, dtype=wp.vec3, requires_grad=True, device=device)

    # set boundary conditions
    static_center_point = (
        torch.from_numpy(gaussian_dict["satic_center_point"]).float().to(device)
    )
    max_static_offset = (
        torch.from_numpy(gaussian_dict["max_static_offset"]).float().to(device)
    )
    velocity = torch.zeros_like(static_center_point)
    # mpm_solver.enforce_particle_velocity_translation(static_center_point, max_static_offset, velocity, 
    #                                                  start_time=0, end_time=1000, device=device)

    material_params = {
        "E": 2.0,  # 0.1-200 MPa
        "nu": 0.1,  # > 0.35
        "material": "jelly",
        # "material": "metal",
        # "friction_angle": 25,
        "g": [0.0, 0.0, 0],
        "density": 0.02,  # kg / m^3
    }

    print("pre set")
    mpm_solver.set_parameters_dict(material_params)
    print("set")
    mpm_solver.finalize_mu_lam()  # set mu and lambda from the E and nu input
    print("finalize")

    total_time = 0.1
    time_step = 0.01
    total_iters = int(total_time / time_step)
    total_iters = 3
    loss = torch.zeros(1, device=device)
    loss = wp.from_torch(loss, requires_grad=True)

    E_tensor = (torch.ones(velocity.shape[0]) * 2.0).contiguous().to(device)
    nu_tensor = (torch.ones(velocity.shape[0]) * 0.1).contiguous().to(device)
    E_warp = wp.from_torch(E_tensor, requires_grad=True)
    nu_warp = wp.from_torch(nu_tensor, requires_grad=True)

    mpm_solver.set_require_grad()

    dt = time_step
    # from IPython import embed; embed()
    with tape:
        # mpm_solver.reset_material(E_warp, nu_warp, device=device)
        # for k in tqdm(range(1, total_iters)):
        # mpm_solver.p2g2p(k, time_step, device=device)

        wp.launch(
            kernel=g2p_test,
            dim=mpm_solver.n_particles,
            inputs=[mpm_solver.mpm_state, mpm_solver.mpm_model, dt],
            device=device,
        )  # x, v, C, F_trial are updated
        # wp.launch(position_loss_kernel, dim=mpm_solver.n_particles,  inputs=[mpm_solver.mpm_state, loss], device=device)
        for i in range(2):
            # wp.launch(position_loss_kernel, dim=mpm_solver.n_particles,  inputs=[mpm_solver.mpm_state, loss], device=device)
            wp.launch(position_loss_kernel, dim=mpm_solver.n_particles,  inputs=[mpm_state, loss], device=device)
            # wp.launch(position_loss_kernel_raw, dim=mpm_solver.n_particles,  inputs=[mpm_state.particle_x, loss], device=device)
        
    tape.backward(loss) # 75120.86
    
    print(loss)
    # model_grad = tape.gradients[mpm_solver.mpm_model]
    # state_grad = tape.gradients[mpm_solver.mpm_state]
    # v_grad = state_grad.particle_v
    # x_grad = state_grad.particle_x
    v_grad = mpm_solver.mpm_state.particle_v.grad
    x_grad = mpm_solver.mpm_state.particle_x.grad
    # E_grad = wp.to_torch(tape.gradients[E_warp])
    print(x_grad)
    from IPython import embed; embed()


@wp.kernel
def g2p_test(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = new_v + grid_v * weight
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        state.particle_v[p] = new_v
        # state.particle_x[p] = state.particle_x[p] + dt * new_v
        # state.particle_x[p] = state.particle_x[p] + dt * state.particle_v[p]
        wp.atomic_add(state.particle_x, p, dt * state.particle_v[p])
        state.particle_C[p] = new_C
        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        F_tmp = (I33 + new_F * dt) * state.particle_F[p]
        state.particle_F_trial[p] = F_tmp

        if model.update_cov_with_F:
            update_cov(state, p, new_F, dt)


@wp.kernel
def position_loss_kernel(mpm_state: MPMStateStruct, loss: wp.array(dtype=float)):

    tid = wp.tid()

    pos = mpm_state.particle_x[tid]
    wp.atomic_add(loss, 0, pos[0] + pos[1] + pos[2])
    # wp.atomic_add(loss, 0, mpm_state.particle_x[tid][0])

@wp.kernel
def position_loss_kernel_raw(particle_x: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):

    tid = wp.tid()

    pos = particle_x[tid]
    wp.atomic_add(loss, 0, pos[0] + pos[1] + pos[2])
    # wp.atomic_add(loss, 0, mpm_state.particle_x[tid][0])


if __name__ == "__main__":
    Fire(test)
