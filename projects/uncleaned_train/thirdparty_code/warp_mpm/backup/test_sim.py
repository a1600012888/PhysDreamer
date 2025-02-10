import warp as wp 
import numpy as np
import torch

@wp.struct
class MPMStateStruct:
    ###### essential #####
    # particle
    particle_x: wp.array(dtype=wp.vec3)  # current position
    particle_v: wp.array(dtype=wp.vec3)  # particle velocity

    particle_vol: wp.array(dtype=float)  # current volume
    particle_F: wp.array(dtype=wp.mat33)  # particle elastic deformation gradient
    grid_v_out: wp.array(
        dtype=wp.vec3, ndim=3
    )  # grid node momentum/velocity, after grid update

class MPM_Simulator_WARPDiff:
    def __init__(self, x, v, vol, device):

        self.mpm_state = MPMStateStruct()
        self.mpm_state.particle_x = wp.array(x, dtype=wp.vec3, requires_grad=True, device=device)
        self.mpm_state.particle_v = wp.array(v, dtype=wp.vec3, requires_grad=True, device=device)
        self.mpm_state.particle_vol = wp.array(vol, dtype=float, requires_grad=False, device=device)
        self.mpm_state.particle_F = wp.array(np.zeros((100, 3, 3), dtype=np.float32), dtype=wp.mat33, requires_grad=True, device=device)
        self.mpm_state.grid_v_out = wp.array(np.zeros((100, 100, 100, 3), dtype=np.float32), dtype=wp.vec3, requires_grad=True, device=device)


@wp.kernel
def vec3_add(mpm_state: MPMStateStruct, selection: wp.array(dtype=wp.float32), dt: float):

    tid = wp.tid()

    # new_v = wp.vec3(1.0, 1.0, 1.0)
    # velocity[tid] = new_v
    velocity = mpm_state.particle_v
    if selection[tid] == 0: # no problem. static condition/loop no problem!
        for i in range(2):
            for j in range(2):
                # x[tid] = x[tid] + velocity[tid] * dt
                # x[tid] = wp.add(x[tid], velocity[tid]) # same as above. wrong gradient
                wp.atomic_add(mpm_state.particle_x, tid, velocity[tid] * mpm_state.particle_vol[tid])

                # x[tid] += velocity[tid] * dt # error, no support of +=
    
@wp.kernel
def loss_kernel(mpm_state: MPMStateStruct,  loss: wp.array(dtype=float)):

    tid = wp.tid()

    pos = mpm_state.particle_x[tid]

    wp.atomic_add(loss, 0, pos[0])


def main():

    wp.init()
    wp.config.verify_cuda = True

    device = 0
    device = "cuda:{}".format(device)


    x = np.random.rand(100, 3).astype(np.float32)
    velocity = np.random.rand(100, 3).astype(np.float32)
    dt = 0.1

    
    # mpm_state = MPMStateStruct()
    # mpm_state.particle_x = wp.array(x, device=device, dtype=wp.vec3,  requires_grad=True)
    # mpm_state.particle_v = wp.array(velocity, device=device, dtype=wp.vec3, requires_grad=True)
    # mpm_state.particle_vol = wp.full(shape=100, value=1, device=device, dtype=wp.float32, requires_grad=False)
    
    mpm_solver = MPM_Simulator_WARPDiff(x, velocity, np.ones(100, dtype=np.float32), device=device)
    
    selection = wp.zeros(100, device=device, dtype=wp.float32)

    loss = torch.zeros(1, device=device)
    loss = wp.from_torch(loss, requires_grad=True)
    tape = wp.Tape()

    with tape:
        for j in range(2):
            wp.launch(vec3_add, dim=100, inputs=[mpm_solver.mpm_state, selection, dt], device=device)
        wp.launch(loss_kernel, dim=100, inputs=[mpm_solver.mpm_state, loss], device=device)

    tape.backward(loss) 

    v_grad = mpm_solver.mpm_state.particle_v.grad
    x_grad = mpm_solver.mpm_state.particle_x.grad
    print(v_grad, loss)

    from IPython import embed; embed()

if __name__ == "__main__":  
    main()