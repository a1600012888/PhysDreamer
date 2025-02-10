
import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
import torch
wp.init()
wp.config.verify_cuda = True


dvc = "cuda:0"

mpm_solver = MPM_Simulator_WARP(10) # initialize with whatever number is fine. it will be reintialized


# You can either load sampling data from an external h5 file, containing initial position (n,3) and particle_volume (n,)
mpm_solver.load_from_sampling("sand_column.h5", n_grid = 150, device=dvc)

# Or load from torch tensor (also position and volume)
# Here we borrow the data from h5, but you can use your own
# [N]
volume_tensor = torch.ones(mpm_solver.n_particles) * 2.5e-8

# torch.float32, [N, 3]
position_tensor = mpm_solver.export_particle_x_to_torch()
print(position_tensor.max(), position_tensor.min())

mpm_solver.load_initial_data_from_torch(position_tensor, volume_tensor)
print(position_tensor.shape, position_tensor.dtype, volume_tensor.shape, volume_tensor.dtype)

# Note: You must provide 'density=..' to set particle_mass = density * particle_volume

material_params = {
    'E': 2000,
    'nu': 0.2,
    "material": "sand",
    'friction_angle': 35,
    'g': [0.0, 0.0, -4.0],
    "density": 200.0
}
mpm_solver.set_parameters_dict(material_params)

mpm_solver.finalize_mu_lam() # set mu and lambda from the E and nu input

mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0,0.0,1.0), 'sticky', 0.0)

from IPython import embed
# embed()

directory_to_save = './sim_results'

save_data_at_frame(mpm_solver, directory_to_save, 0, save_to_ply=True, save_to_h5=False)

for k in range(1,50):
    mpm_solver.p2g2p(k, 0.002, device=dvc)
    save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=False)



# extract the position, make some changes, load it back
position = mpm_solver.export_particle_x_to_torch()
# e.g. we shift the x position
position[:,0] = position[:,0] + 0.1
mpm_solver.import_particle_x_from_torch(position)
# keep running sim
for k in range(50,100):

    mpm_solver.p2g2p(k, 0.002, device=dvc)
    save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=False)
