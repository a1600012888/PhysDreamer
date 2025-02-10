import warp as wp
import warp.torch
import torch
from typing import Optional, Union, Sequence, Any
from torch import Tensor
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from warp_utils import from_torch_safe


@wp.struct
class MPMStateStruct(object):
    ###### essential #####
    # particle
    particle_x: wp.array(dtype=wp.vec3)  # current position
    particle_v: wp.array(dtype=wp.vec3)  # particle velocity
    particle_F: wp.array(dtype=wp.mat33)  # particle elastic deformation gradient
    particle_cov: wp.array(dtype=float)  # current covariance matrix
    particle_F_trial: wp.array(
        dtype=wp.mat33
    )  # apply return mapping on this to obtain elastic def grad
    particle_stress: wp.array(dtype=wp.mat33)  # Kirchoff stress, elastic stress
    particle_C: wp.array(dtype=wp.mat33)
    particle_vol: wp.array(dtype=float)  # current volume
    particle_mass: wp.array(dtype=float)  # mass
    particle_density: wp.array(dtype=float)  # density

    particle_selection: wp.array(
        dtype=int
    )  # only particle_selection[p] = 0 will be simulated

    # grid
    grid_m: wp.array(dtype=float, ndim=3)
    grid_v_in: wp.array(dtype=wp.vec3, ndim=3)  # grid node momentum/velocity
    grid_v_out: wp.array(
        dtype=wp.vec3, ndim=3
    )  # grid node momentum/velocity, after grid update

    def init(
        self,
        shape: Union[Sequence[int], int],
        device: wp.context.Devicelike = None,
        requires_grad=False,
    ) -> None:
        # shape default is int. number of particles
        self.particle_x = wp.zeros(
            shape, dtype=wp.vec3, device=device, requires_grad=requires_grad
        )
        self.particle_v = wp.zeros(
            shape, dtype=wp.vec3, device=device, requires_grad=requires_grad
        )
        self.particle_F = wp.zeros(
            shape, dtype=wp.mat33, device=device, requires_grad=requires_grad
        )
        self.particle_cov = wp.zeros(
            shape * 6, dtype=float, device=device, requires_grad=False
        )

        self.particle_F_trial = wp.zeros(
            shape, dtype=wp.mat33, device=device, requires_grad=requires_grad
        )

        self.particle_stress = wp.zeros(
            shape, dtype=wp.mat33, device=device, requires_grad=requires_grad
        )
        self.particle_C = wp.zeros(
            shape, dtype=wp.mat33, device=device, requires_grad=requires_grad
        )

        self.particle_vol = wp.zeros(
            shape, dtype=float, device=device, requires_grad=False
        )
        self.particle_mass = wp.zeros(
            shape, dtype=float, device=device, requires_grad=False
        )
        self.particle_density = wp.zeros(
            shape, dtype=float, device=device, requires_grad=False
        )

        self.particle_selection = wp.zeros(
            shape, dtype=int, device=device, requires_grad=False
        )

        # grid: will init later
        self.grid_m = wp.zeros(
            (10, 10, 10), dtype=float, device=device, requires_grad=requires_grad
        )
        self.grid_v_in = wp.zeros(
            (10, 10, 10), dtype=wp.vec3, device=device, requires_grad=requires_grad
        )
        self.grid_v_out = wp.zeros(
            (10, 10, 10), dtype=wp.vec3, device=device, requires_grad=requires_grad
        )

    def init_grid(
        self, grid_res: int, device: wp.context.Devicelike = None, requires_grad=False
    ):
        self.grid_m = wp.zeros(
            (grid_res, grid_res, grid_res),
            dtype=float,
            device=device,
            requires_grad=False,
        )
        self.grid_v_in = wp.zeros(
            (grid_res, grid_res, grid_res),
            dtype=wp.vec3,
            device=device,
            requires_grad=requires_grad,
        )
        self.grid_v_out = wp.zeros(
            (grid_res, grid_res, grid_res),
            dtype=wp.vec3,
            device=device,
            requires_grad=requires_grad,
        )

    def from_torch(
        self,
        tensor_x: Tensor,
        tensor_volume: Tensor,
        tensor_cov: Optional[Tensor] = None,
        tensor_velocity: Optional[Tensor] = None,
        n_grid: int = 100,
        grid_lim=1.0,
        device="cuda:0",
        requires_grad=True,
    ):
        num_dim, n_particles = tensor_x.shape[1], tensor_x.shape[0]
        assert tensor_x.shape[0] == tensor_volume.shape[0]
        # assert tensor_x.shape[0] == tensor_cov.reshape(-1, 6).shape[0]
        self.init_grid(grid_res=n_grid, device=device, requires_grad=requires_grad)

        if tensor_x is not None:
            self.particle_x = from_torch_safe(
                tensor_x.contiguous().detach().clone(),
                dtype=wp.vec3,
                requires_grad=requires_grad,
            )

        if tensor_volume is not None:
            print(self.particle_vol.shape, tensor_volume.shape)
            volume_numpy = tensor_volume.detach().cpu().numpy()
            self.particle_vol = wp.from_numpy(
                volume_numpy, dtype=float, device=device, requires_grad=False
            )

        if tensor_cov is not None:
            cov_numpy = tensor_cov.reshape(-1).detach().clone().cpu().numpy()
            self.particle_cov = wp.from_numpy(
                cov_numpy, dtype=float, device=device, requires_grad=False
            )

        if tensor_velocity is not None:
            self.particle_v = from_torch_safe(
                tensor_velocity.contiguous().detach().clone(),
                dtype=wp.vec3,
                requires_grad=requires_grad,
            )

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=n_particles,
            inputs=[self.particle_F_trial],
            device=device,
        )
        # initial trial deformation gradient is set to identity

        print("Particles initialized from torch data.")
        print("Total particles: ", n_particles)

    def reset_state(
        self,
        tensor_x: Tensor,
        tensor_cov: Optional[Tensor] = None,
        tensor_velocity: Optional[Tensor] = None,
        tensor_density: Optional[Tensor] = None,
        selection_mask: Optional[Tensor] = None,
        device="cuda:0",
        requires_grad=True,
    ):
        # reset p_c, p_v, p_C, p_F_trial
        num_dim, n_particles = tensor_x.shape[1], tensor_x.shape[0]

        # assert tensor_x.shape[0] == tensor_cov.reshape(-1, 6).shape[0]

        if tensor_x is not None:
            self.particle_x = from_torch_safe(
                tensor_x.contiguous().detach(),
                dtype=wp.vec3,
                requires_grad=requires_grad,
            )

        if tensor_cov is not None:
            cov_numpy = tensor_cov.reshape(-1).detach().clone().cpu().numpy()
            self.particle_cov = wp.from_numpy(
                cov_numpy, dtype=float, device=device, requires_grad=False
            )

        if tensor_velocity is not None:
            self.particle_v = from_torch_safe(
                tensor_velocity.contiguous().detach().clone(),
                dtype=wp.vec3,
                requires_grad=requires_grad,
            )

        if tensor_density is not None and selection_mask is not None:
            wp_density = from_torch_safe(
                tensor_density.contiguous().detach().clone(),
                dtype=wp.float32,
                requires_grad=False,
            )
            # 1 indicate we need to simulate this particle
            wp_selection_mask = from_torch_safe(
                selection_mask.contiguous().detach().clone().type(torch.int),
                dtype=wp.int32,
                requires_grad=False,
            )

            wp.launch(
                kernel=set_float_vec_to_vec_wmask,
                dim=n_particles,
                inputs=[self.particle_density, wp_density, wp_selection_mask],
                device=device,
            )

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=n_particles,
            inputs=[self.particle_F_trial],
            device=device,
        )
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=n_particles,
            inputs=[self.particle_F],
            device=device,
        )

        wp.launch(
            kernel=set_mat33_to_zero,
            dim=n_particles,
            inputs=[self.particle_C],
            device=device,
        )

        wp.launch(
            kernel=set_mat33_to_zero,
            dim=n_particles,
            inputs=[self.particle_stress],
            device=device,
        )

    def continue_from_torch(
        self,
        tensor_x: Tensor,
        tensor_velocity: Optional[Tensor] = None,
        tensor_F: Optional[Tensor] = None,
        tensor_C: Optional[Tensor] = None,
        device="cuda:0",
        requires_grad=True,
    ):
        if tensor_x is not None:
            self.particle_x = from_torch_safe(
                tensor_x.contiguous().detach(),
                dtype=wp.vec3,
                requires_grad=requires_grad,
            )

        if tensor_velocity is not None:
            self.particle_v = from_torch_safe(
                tensor_velocity.contiguous().detach().clone(),
                dtype=wp.vec3,
                requires_grad=requires_grad,
            )

        if tensor_F is not None:
            self.particle_F_trial = from_torch_safe(
                tensor_F.contiguous().detach().clone(),
                dtype=wp.mat33,
                requires_grad=requires_grad,
            )

        if tensor_C is not None:
            self.particle_C = from_torch_safe(
                tensor_C.contiguous().detach().clone(),
                dtype=wp.mat33,
                requires_grad=requires_grad,
            )

    def set_require_grad(self, requires_grad=True):
        self.particle_x.requires_grad = requires_grad
        self.particle_v.requires_grad = requires_grad
        self.particle_F.requires_grad = requires_grad
        self.particle_F_trial.requires_grad = requires_grad
        self.particle_stress.requires_grad = requires_grad
        self.particle_C.requires_grad = requires_grad

        self.grid_v_out.requires_grad = requires_grad
        self.grid_v_in.requires_grad = requires_grad

    def reset_density(
        self,
        tensor_density: Tensor,
        selection_mask: Optional[Tensor] = None,
        device="cuda:0",
        requires_grad=True,
        update_mass=False,
    ):
        n_particles = tensor_density.shape[0]
        if tensor_density is not None:
            wp_density = from_torch_safe(
                tensor_density.contiguous().detach().clone(),
                dtype=wp.float32,
                requires_grad=False,
            )
        
        if selection_mask is not None:
            # 1 indicate we need to simulate this particle
            wp_selection_mask = from_torch_safe(
                selection_mask.contiguous().detach().clone().type(torch.int),
                dtype=wp.int32,
                requires_grad=False,
            )

            wp.launch(
                kernel=set_float_vec_to_vec_wmask,
                dim=n_particles,
                inputs=[self.particle_density, wp_density, wp_selection_mask],
                device=device,
            )
        else:
            wp.launch(
                kernel=set_float_vec_to_vec,
                dim=n_particles,
                inputs=[self.particle_density, wp_density],
                device=device,
            )

        if update_mass:
            num_particles = self.particle_x.shape[0]
            wp.launch(
                kernel=get_float_array_product,
                dim=num_particles,
                inputs=[
                    self.particle_density,
                    self.particle_vol,
                    self.particle_mass,
                ],
                device=device,
            )

    def partial_clone(self, device="cuda:0", requires_grad=True):
        new_state = MPMStateStruct()
        n_particles = self.particle_x.shape[0]
        new_state.init(n_particles, device=device, requires_grad=requires_grad)

        # clone section:
        # new_state.particle_vol = wp.clone(self.particle_vol, requires_grad=False)
        # new_state.particle_density = wp.clone(self.particle_density, requires_grad=False)
        # new_state.particle_mass = wp.clone(self.particle_mass, requires_grad=False)

        # new_state.particle_selection = wp.clone(self.particle_selection, requires_grad=False)

        wp.copy(new_state.particle_vol, self.particle_vol)
        wp.copy(new_state.particle_density, self.particle_density)
        wp.copy(new_state.particle_mass, self.particle_mass)
        wp.copy(new_state.particle_selection, self.particle_selection)

        # init grid to zero with grid res.
        new_state.init_grid(
            grid_res=self.grid_v_in.shape[0], device=device, requires_grad=requires_grad
        )

        # init some matrix to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=n_particles,
            inputs=[new_state.particle_F_trial],
            device=device,
        )

        new_state.set_require_grad(requires_grad=requires_grad)
        return new_state


@wp.struct
class MPMModelStruct(object):
    ####### essential #######
    grid_lim: float
    n_particles: int
    n_grid: int
    dx: float
    inv_dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    mu: wp.array(dtype=float)
    lam: wp.array(dtype=float)
    E: wp.array(dtype=float)
    nu: wp.array(dtype=float)
    material: int

    ######## for plasticity ####
    yield_stress: wp.array(dtype=float)
    friction_angle: float
    alpha: float
    gravitational_accelaration: wp.vec3
    hardening: float
    xi: float
    plastic_viscosity: float
    softening: float

    ####### for damping
    rpic_damping: float
    grid_v_damping_scale: float

    ####### for PhysGaussian: covariance
    update_cov_with_F: int

    def init(
        self,
        shape: Union[Sequence[int], int],
        device: wp.context.Devicelike = None,
        requires_grad=False,
    ) -> None:
        self.E = wp.zeros(
            shape, dtype=float, device=device, requires_grad=requires_grad
        )  # young's modulus
        self.nu = wp.zeros(
            shape, dtype=float, device=device, requires_grad=requires_grad
        )  # poisson's ratio

        self.mu = wp.zeros(
            shape, dtype=float, device=device, requires_grad=requires_grad
        )
        self.lam = wp.zeros(
            shape, dtype=float, device=device, requires_grad=requires_grad
        )

        self.yield_stress = wp.zeros(
            shape, dtype=float, device=device, requires_grad=requires_grad
        )

    def finalize_mu_lam(self, n_particles, device="cuda:0"):
        wp.launch(
            kernel=compute_mu_lam_from_E_nu_clean,
            dim=n_particles,
            inputs=[self.mu, self.lam, self.E, self.nu],
            device=device,
        )

    def init_other_params(self, n_grid=100, grid_lim=1.0, device="cuda:0"):
        self.grid_lim = grid_lim
        self.n_grid = n_grid
        self.grid_dim_x = n_grid
        self.grid_dim_y = n_grid
        self.grid_dim_z = n_grid
        (
            self.dx,
            self.inv_dx,
        ) = self.grid_lim / self.n_grid, float(
            n_grid / grid_lim
        )  # [0-1]?

        self.update_cov_with_F = False

        # material is used to switch between different elastoplastic models. 0 is jelly
        self.material = 0

        self.plastic_viscosity = 0.0
        self.softening = 0.1
        self.friction_angle = 25.0
        sin_phi = wp.sin(self.friction_angle / 180.0 * 3.14159265)
        self.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        self.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)

        self.rpic_damping = 0.0  # 0.0 if no damping (apic). -1 if pic

        self.grid_v_damping_scale = 1.1  # globally applied

    def from_torch(
        self, tensor_E: Tensor, tensor_nu: Tensor, device="cuda:0", requires_grad=False
    ):
        self.E = wp.from_torch(tensor_E.contiguous(), requires_grad=requires_grad)
        self.nu = wp.from_torch(tensor_nu.contiguous(), requires_grad=requires_grad)
        n_particles = tensor_E.shape[0]
        self.finalize_mu_lam(n_particles=n_particles, device=device)

    def set_require_grad(self, requires_grad=True):
        self.E.requires_grad = requires_grad
        self.nu.requires_grad = requires_grad
        self.mu.requires_grad = requires_grad
        self.lam.requires_grad = requires_grad


# for various boundary conditions
@wp.struct
class Dirichlet_collider:
    point: wp.vec3
    normal: wp.vec3
    direction: wp.vec3

    start_time: float
    end_time: float

    friction: float
    surface_type: int

    velocity: wp.vec3

    threshold: float
    reset: int
    index: int

    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    height: float
    length: float
    R: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3
    half_height_and_radius: wp.vec2


@wp.struct
class GridCollider:
    point: wp.vec3
    normal: wp.vec3
    direction: wp.vec3

    start_time: float
    end_time: float
    mask: wp.array(dtype=int, ndim=3)


@wp.struct
class Impulse_modifier:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    force: wp.vec3
    forceTimesDt: wp.vec3
    numsteps: int

    point: wp.vec3
    size: wp.vec3
    mask: wp.array(dtype=int)


@wp.struct
class MPMtailoredStruct:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    friction: float
    surface_type: int
    velocity: wp.vec3
    threshold: float
    reset: int

    point_rotate: wp.vec3
    normal_rotate: wp.vec3
    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    point_plane: wp.vec3
    normal_plane: wp.vec3
    velocity_plane: wp.vec3
    threshold_plane: float


@wp.struct
class MaterialParamsModifier:
    point: wp.vec3
    size: wp.vec3
    E: float
    nu: float
    density: float


@wp.struct
class ParticleVelocityModifier:
    point: wp.vec3
    normal: wp.vec3
    half_height_and_radius: wp.vec2
    rotation_scale: float
    translation_scale: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3

    start_time: float

    end_time: float

    velocity: wp.vec3

    mask: wp.array(dtype=int)


@wp.kernel
def compute_mu_lam_from_E_nu_clean(
    mu: wp.array(dtype=float),
    lam: wp.array(dtype=float),
    E: wp.array(dtype=float),
    nu: wp.array(dtype=float),
):
    p = wp.tid()
    mu[p] = E[p] / (2.0 * (1.0 + nu[p]))
    lam[p] = E[p] * nu[p] / ((1.0 + nu[p]) * (1.0 - 2.0 * nu[p]))


@wp.kernel
def set_vec3_to_zero(target_array: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    target_array[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def set_vec3_to_vec3(
    source_array: wp.array(dtype=wp.vec3), target_array: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    source_array[tid] = target_array[tid]


@wp.kernel
def set_float_vec_to_vec_wmask(
    source_array: wp.array(dtype=float),
    target_array: wp.array(dtype=float),
    selection_mask: wp.array(dtype=int),
):
    tid = wp.tid()
    if selection_mask[tid] == 1:
        source_array[tid] = target_array[tid]


@wp.kernel
def set_float_vec_to_vec(
    source_array: wp.array(dtype=float), target_array: wp.array(dtype=float)
):
    tid = wp.tid()
    source_array[tid] = target_array[tid]


@wp.kernel
def set_mat33_to_identity(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@wp.kernel
def set_mat33_to_zero(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def add_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.add(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def subtract_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.sub(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def add_vec3_to_vec3(
    first_array: wp.array(dtype=wp.vec3), second_array: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    first_array[tid] = wp.add(first_array[tid], second_array[tid])


@wp.kernel
def set_value_to_float_array(target_array: wp.array(dtype=float), value: float):
    tid = wp.tid()
    target_array[tid] = value


@wp.kernel
def set_warpvalue_to_float_array(
    target_array: wp.array(dtype=float), value: warp.types.float32
):
    tid = wp.tid()
    target_array[tid] = value


@wp.kernel
def get_float_array_product(
    arrayA: wp.array(dtype=float),
    arrayB: wp.array(dtype=float),
    arrayC: wp.array(dtype=float),
):
    tid = wp.tid()
    arrayC[tid] = arrayA[tid] * arrayB[tid]


def torch2warp_quat(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 4
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.quat,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_float(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=warp.types.float32,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_vec3(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.vec3,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_mat33(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.mat33,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a
