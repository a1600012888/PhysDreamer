import sys
import os

import warp as wp

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from mpm_data_structure import *
from mpm_utils import *
from typing import Optional, Union, Sequence, Any


class MPMWARPDiff(object):
    # def __init__(self, n_particles, n_grid=100, grid_lim=1.0, device="cuda:0"):
    #     self.initialize(n_particles, n_grid, grid_lim, device=device)
    #     self.time_profile = {}

    def __init__(self, n_particles, n_grid=100, grid_lim=1.0, device="cuda:0"):
        self.initialize(n_particles, n_grid, grid_lim, device=device)
        self.time_profile = {}

    def initialize(self, n_particles, n_grid=100, grid_lim=1.0, device="cuda:0"):
        self.n_particles = n_particles

        self.time = 0.0

        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []

        self.tailored_struct_for_bc = MPMtailoredStruct()
        self.pre_p2g_operations = []
        self.impulse_params = []

        self.particle_velocity_modifiers = []
        self.particle_velocity_modifier_params = []

    # must give density. mass will be updated as density * volume
    def set_parameters(self, device="cuda:0", **kwargs):
        self.set_parameters_dict(device, kwargs)

    def set_parameters_dict(self, mpm_model, mpm_state, kwargs={}, device="cuda:0"):
        if "material" in kwargs:
            if kwargs["material"] == "jelly":
                mpm_model.material = 0
            elif kwargs["material"] == "metal":
                mpm_model.material = 1
            elif kwargs["material"] == "sand":
                mpm_model.material = 2
            elif kwargs["material"] == "foam":
                mpm_model.material = 3
            elif kwargs["material"] == "snow":
                mpm_model.material = 4
            elif kwargs["material"] == "plasticine":
                mpm_model.material = 5
            else:
                raise TypeError("Undefined material type")

        if "yield_stress" in kwargs:
            val = kwargs["yield_stress"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[mpm_model.yield_stress, val],
                device=device,
            )
        if "hardening" in kwargs:
            mpm_model.hardening = kwargs["hardening"]
        if "xi" in kwargs:
            mpm_model.xi = kwargs["xi"]
        if "friction_angle" in kwargs:
            mpm_model.friction_angle = kwargs["friction_angle"]
            sin_phi = wp.sin(mpm_model.friction_angle / 180.0 * 3.14159265)
            mpm_model.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        if "g" in kwargs:
            mpm_model.gravitational_accelaration = wp.vec3(
                kwargs["g"][0], kwargs["g"][1], kwargs["g"][2]
            )

        if "density" in kwargs:
            density_value = kwargs["density"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[mpm_state.particle_density, density_value],
                device=device,
            )
            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    mpm_state.particle_density,
                    mpm_state.particle_vol,
                    mpm_state.particle_mass,
                ],
                device=device,
            )
        if "rpic_damping" in kwargs:
            mpm_model.rpic_damping = kwargs["rpic_damping"]
        if "plastic_viscosity" in kwargs:
            mpm_model.plastic_viscosity = kwargs["plastic_viscosity"]
        if "softening" in kwargs:
            mpm_model.softening = kwargs["softening"]
        if "grid_v_damping_scale" in kwargs:
            mpm_model.grid_v_damping_scale = kwargs["grid_v_damping_scale"]

    def set_E_nu(self, mpm_model, E: float, nu: float, device="cuda:0"):

        if isinstance(E, float):
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[mpm_model.E, E],
                device=device,
            )
        else: # E is warp array
            wp.launch(
                kernel=set_float_vec_to_vec,
                dim=self.n_particles,
                inputs=[mpm_model.E, E],
                device=device,
            ) 

        if isinstance(nu, float):
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[mpm_model.nu, nu],
                device=device,
            )
        else:
            wp.launch(
                kernel=set_float_vec_to_vec,
                dim=self.n_particles,
                inputs=[mpm_model.nu, nu],
                device=device,
            )

    def p2g2p(self, mpm_model, mpm_state, step, dt, device="cuda:0"):
        grid_size = (
            mpm_model.grid_dim_x,
            mpm_model.grid_dim_y,
            mpm_model.grid_dim_z,
        )

        # TODO, move this outside of the loop!!
        wp.launch(
            kernel=compute_mu_lam_from_E_nu,
            dim=self.n_particles,
            inputs=[mpm_state, mpm_model],
            device=device,
        )
        wp.launch(
            kernel=zero_grid,  # gradient might gone
            dim=(grid_size),
            inputs=[mpm_state, mpm_model],
            device=device,
        )

        # apply pre-p2g operations on particles
        # apply impulse force on particles..
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, mpm_state, self.impulse_params[k]],
                device=device,
            )

        # apply dirichlet particle v modifier
        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(
                kernel=self.particle_velocity_modifiers[k],
                dim=self.n_particles,
                inputs=[
                    self.time,
                    mpm_state,
                    self.particle_velocity_modifier_params[k],
                ],
                device=device,
            )

        # compute stress = stress(returnMap(F_trial))
        # F_trail => F                    # TODO: this is overite.. 
        # F, SVD(F), lam, mu => Stress.   # TODO: this is overite.. 
            
        with wp.ScopedTimer(
            "compute_stress_from_F_trial",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_stress_from_F_trial,
                dim=self.n_particles,
                inputs=[mpm_state, mpm_model, dt],
                device=device,
            )  # F and stress are updated

        # p2g
        with wp.ScopedTimer(
            "p2g",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=p2g_apic_with_stress,
                dim=self.n_particles,
                inputs=[mpm_state, mpm_model, dt],
                device=device,
            )  # apply p2g'

        # grid update
        with wp.ScopedTimer(
            "grid_update", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=grid_normalization_and_gravity,
                dim=(grid_size),
                inputs=[mpm_state, mpm_model, dt],
                device=device,
            )

        if mpm_model.grid_v_damping_scale < 1.0:
            wp.launch(
                kernel=add_damping_via_grid,
                dim=(grid_size),
                inputs=[mpm_state, mpm_model.grid_v_damping_scale],
                device=device,
            )

        # apply BC on grid, collide
        with wp.ScopedTimer(
            "apply_BC_on_grid", synchronize=True, print=False, dict=self.time_profile
        ):
            for k in range(len(self.grid_postprocess)):
                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=grid_size,
                    inputs=[
                        self.time,
                        dt,
                        mpm_state,
                        mpm_model,
                        self.collider_params[k],
                    ],
                    device=device,
                )
                if self.modify_bc[k] is not None:
                    self.modify_bc[k](self.time, dt, self.collider_params[k])

        # g2p
        with wp.ScopedTimer(
            "g2p", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=g2p,
                dim=self.n_particles,
                inputs=[mpm_state, mpm_model, dt],
                device=device,
            )  # x, v, C, F_trial are updated

        #### CFL check ####
        # particle_v = self.mpm_state.particle_v.numpy()
        # if np.max(np.abs(particle_v)) > self.mpm_model.dx / dt:
        #     print("max particle v: ", np.max(np.abs(particle_v)))
        #     print("max allowed  v: ", self.mpm_model.dx / dt)
        #     print("does not allow v*dt>dx")
        #     input()
        #### CFL check ####
        with wp.ScopedTimer(
            "clip_particle_x", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=clip_particle_x,
                dim=self.n_particles,
                inputs=[mpm_state, mpm_model],
                device=device,
            )

        self.time = self.time + dt

    def print_time_profile(self):
        print("MPM Time profile:")
        for key, value in self.time_profile.items():
            print(key, sum(value))

    # a surface specified by a point and the normal vector
    def add_surface_collider(
        self,
        point,
        normal,
        surface="sticky",
        friction=0.0,
        start_time=0.0,
        end_time=999.0,
    ):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / wp.sqrt(float(sum(x**2 for x in normal)))
        normal = list(normal_scale * x for x in normal)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])

        if surface == "sticky" and friction != 0:
            raise ValueError("friction must be 0 on sticky surfaces.")
        if surface == "sticky":
            collider_param.surface_type = 0
        elif surface == "slip":
            collider_param.surface_type = 1
        elif surface == "cut":
            collider_param.surface_type = 11
        else:
            collider_param.surface_type = 2
        # frictional
        collider_param.friction = friction

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
                dotproduct = wp.dot(offset, n)

                if dotproduct < 0.0:
                    if param.surface_type == 0:
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )
                    elif param.surface_type == 11:
                        if (
                            float(grid_z) * model.dx < 0.4
                            or float(grid_z) * model.dx > 0.53
                        ):
                            state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                                0.0, 0.0, 0.0
                            )
                        else:
                            v_in = state.grid_v_out[grid_x, grid_y, grid_z]
                            state.grid_v_out[grid_x, grid_y, grid_z] = (
                                wp.vec3(v_in[0], 0.0, v_in[2]) * 0.3
                            )
                    else:
                        v = state.grid_v_out[grid_x, grid_y, grid_z]
                        normal_component = wp.dot(v, n)
                        if param.surface_type == 1:
                            v = (
                                v - normal_component * n
                            )  # Project out all normal component
                        else:
                            v = (
                                v - wp.min(normal_component, 0.0) * n
                            )  # Project out only inward normal component
                        if normal_component < 0.0 and wp.length(v) > 1e-20:
                            v = wp.max(
                                0.0, wp.length(v) + normal_component * param.friction
                            ) * wp.normalize(
                                v
                            )  # apply friction here
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    # a cubiod is a rectangular cube'
    # centered at `point`
    # dimension is x: point[0]±size[0]
    #              y: point[1]±size[1]
    #              z: point[2]±size[2]
    # all grid nodes lie within the cubiod will have their speed set to velocity
    # the cuboid itself is also moving with const speed = velocity
    # set the speed to zero to fix BC
    def set_velocity_on_cuboid(
        self,
        point,
        size,
        velocity,
        start_time=0.0,
        end_time=999.0,
        reset=0,
    ):
        point = list(point)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.size = size
        collider_param.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])
        # collider_param.threshold = threshold
        collider_param.reset = reset
        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                if (
                    wp.abs(offset[0]) < param.size[0]
                    and wp.abs(offset[1]) < param.size[1]
                    and wp.abs(offset[2]) < param.size[2]
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = param.velocity
            elif param.reset == 1:
                if time < param.end_time + 15.0 * dt:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

        def modify(time, dt, param: Dirichlet_collider):
            if time >= param.start_time and time < param.end_time:
                param.point = wp.vec3(
                    param.point[0] + dt * param.velocity[0],
                    param.point[1] + dt * param.velocity[1],
                    param.point[2] + dt * param.velocity[2],
                )  # param.point + dt * param.velocity

        self.grid_postprocess.append(collide)
        self.modify_bc.append(modify)

    def add_bounding_box(self, start_time=0.0, end_time=999.0):
        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            padding = 3
            if time >= param.start_time and time < param.end_time:
                if grid_x < padding and state.grid_v_out[grid_x, grid_y, grid_z][0] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )
                if (
                    grid_x >= model.grid_dim_x - padding
                    and state.grid_v_out[grid_x, grid_y, grid_z][0] > 0
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )

                if grid_y < padding and state.grid_v_out[grid_x, grid_y, grid_z][1] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )
                if (
                    grid_y >= model.grid_dim_y - padding
                    and state.grid_v_out[grid_x, grid_y, grid_z][1] > 0
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )

                if grid_z < padding and state.grid_v_out[grid_x, grid_y, grid_z][2] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        0.0,
                    )
                if (
                    grid_z >= model.grid_dim_z - padding
                    and state.grid_v_out[grid_x, grid_y, grid_z][2] > 0
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        0.0,
                    )

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    # particle_v += force/particle_mass * dt
    # this is applied from start_dt, ends after num_dt p2g2p's
    # particle velocity is changed before p2g at each timestep
    def add_impulse_on_particles(
        self,
        mpm_state,
        force,
        dt,
        point=[1, 1, 1],
        size=[1, 1, 1],
        num_dt=1,
        start_time=0.0,
        device="cuda:0",
    ):
        impulse_param = Impulse_modifier()
        impulse_param.start_time = start_time
        impulse_param.end_time = start_time + dt * num_dt

        impulse_param.point = wp.vec3(point[0], point[1], point[2])
        impulse_param.size = wp.vec3(size[0], size[1], size[2])
        impulse_param.mask = wp.zeros(shape=self.n_particles, dtype=int, device=device)

        impulse_param.force = wp.vec3(
            force[0],
            force[1],
            force[2],
        )

        wp.launch(
            kernel=selection_add_impulse_on_particles,
            dim=self.n_particles,
            inputs=[mpm_state, impulse_param],
            device=device,
        )

        self.impulse_params.append(impulse_param)

        @wp.kernel
        def apply_force(
            time: float, dt: float, state: MPMStateStruct, param: Impulse_modifier
        ):
            p = wp.tid()
            if time >= param.start_time and time < param.end_time:
                if param.mask[p] == 1:
                    impulse = wp.vec3(
                        param.force[0] / state.particle_mass[p],
                        param.force[1] / state.particle_mass[p],
                        param.force[2] / state.particle_mass[p],
                    )
                    state.particle_v[p] = state.particle_v[p] + impulse * dt

        self.pre_p2g_operations.append(apply_force)

    def enforce_particle_velocity_translation(
        self, mpm_state, point, size, velocity, start_time, end_time, device="cuda:0"
    ):
        # first select certain particles based on position

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.size = wp.vec3(size[0], size[1], size[2])

        velocity_modifier_params.velocity = wp.vec3(
            velocity[0], velocity[1], velocity[2]
        )

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device
        )

        wp.launch(
            kernel=selection_enforce_particle_velocity_translation,
            dim=self.n_particles,
            inputs=[mpm_state, velocity_modifier_params],
            device=device,
        )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)

        @wp.kernel
        def modify_particle_v_before_p2g(
            time: float,
            state: MPMStateStruct,
            velocity_modifier_params: ParticleVelocityModifier,
        ):
            p = wp.tid()
            if (
                time >= velocity_modifier_params.start_time
                and time < velocity_modifier_params.end_time
            ):
                if velocity_modifier_params.mask[p] == 1:
                    state.particle_v[p] = velocity_modifier_params.velocity

        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)

    # define a cylinder with center point, half_height, radius, normal
    # particles within the cylinder are rotating along the normal direction
    # may also have a translational velocity along the normal direction
    def enforce_particle_velocity_rotation(
        self,
        mpm_state,
        point,
        normal,
        half_height_and_radius,
        rotation_scale,
        translation_scale,
        start_time,
        end_time,
        device="cuda:0",
    ):
        normal_scale = 1.0 / wp.sqrt(
            float(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        )
        normal = list(normal_scale * x for x in normal)

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.half_height_and_radius = wp.vec2(
            half_height_and_radius[0], half_height_and_radius[1]
        )
        velocity_modifier_params.normal = wp.vec3(normal[0], normal[1], normal[2])

        horizontal_1 = wp.vec3(1.0, 1.0, 1.0)
        if wp.abs(wp.dot(velocity_modifier_params.normal, horizontal_1)) < 0.01:
            horizontal_1 = wp.vec3(0.72, 0.37, -0.67)
        horizontal_1 = (
            horizontal_1
            - wp.dot(horizontal_1, velocity_modifier_params.normal)
            * velocity_modifier_params.normal
        )
        horizontal_1 = horizontal_1 * (1.0 / wp.length(horizontal_1))
        horizontal_2 = wp.cross(horizontal_1, velocity_modifier_params.normal)

        velocity_modifier_params.horizontal_axis_1 = horizontal_1
        velocity_modifier_params.horizontal_axis_2 = horizontal_2

        velocity_modifier_params.rotation_scale = rotation_scale
        velocity_modifier_params.translation_scale = translation_scale

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device
        )

        wp.launch(
            kernel=selection_enforce_particle_velocity_cylinder,
            dim=self.n_particles,
            inputs=[mpm_state, velocity_modifier_params],
            device=device,
        )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)

        @wp.kernel
        def modify_particle_v_before_p2g(
            time: float,
            state: MPMStateStruct,
            velocity_modifier_params: ParticleVelocityModifier,
        ):
            p = wp.tid()
            if (
                time >= velocity_modifier_params.start_time
                and time < velocity_modifier_params.end_time
            ):
                if velocity_modifier_params.mask[p] == 1:
                    offset = state.particle_x[p] - velocity_modifier_params.point
                    horizontal_distance = wp.length(
                        offset
                        - wp.dot(offset, velocity_modifier_params.normal)
                        * velocity_modifier_params.normal
                    )
                    cosine = (
                        wp.dot(offset, velocity_modifier_params.horizontal_axis_1)
                        / horizontal_distance
                    )
                    theta = wp.acos(cosine)
                    if wp.dot(offset, velocity_modifier_params.horizontal_axis_2) > 0:
                        theta = theta
                    else:
                        theta = -theta
                    axis1_scale = (
                        -horizontal_distance
                        * wp.sin(theta)
                        * velocity_modifier_params.rotation_scale
                    )
                    axis2_scale = (
                        horizontal_distance
                        * wp.cos(theta)
                        * velocity_modifier_params.rotation_scale
                    )
                    axis_vertical_scale = translation_scale
                    state.particle_v[p] = (
                        axis1_scale * velocity_modifier_params.horizontal_axis_1
                        + axis2_scale * velocity_modifier_params.horizontal_axis_2
                        + axis_vertical_scale * velocity_modifier_params.normal
                    )

        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)

    # given normal direction, say [0,0,1]
    # gradually release grid velocities from start position to end position
    def release_particles_sequentially(
        self, normal, start_position, end_position, num_layers, start_time, end_time
    ):
        num_layers = 50
        point = [0, 0, 0]
        size = [0, 0, 0]
        axis = -1
        for i in range(3):
            if normal[i] == 0:
                point[i] = 1
                size[i] = 1
            else:
                axis = i
                point[i] = end_position

        half_length_portion = wp.abs(start_position - end_position) / num_layers
        end_time_portion = end_time / num_layers
        for i in range(num_layers):
            size[axis] = half_length_portion * (num_layers - i)
            self.enforce_particle_velocity_translation(
                point=point,
                size=size,
                velocity=[0, 0, 0],
                start_time=start_time,
                end_time=end_time_portion * (i + 1),
            )

    def enforce_particle_velocity_by_mask(
        self,
        mpm_state,
        selection_mask: torch.Tensor,
        velocity,
        start_time,
        end_time,
    ):
        # first select certain particles based on position

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.velocity = wp.vec3(
            velocity[0], velocity[1], velocity[2]
        )

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.from_torch(selection_mask)

        self.particle_velocity_modifier_params.append(velocity_modifier_params)

        @wp.kernel
        def modify_particle_v_before_p2g(
            time: float,
            state: MPMStateStruct,
            velocity_modifier_params: ParticleVelocityModifier,
        ):
            p = wp.tid()
            if (
                time >= velocity_modifier_params.start_time
                and time < velocity_modifier_params.end_time
            ):
                if velocity_modifier_params.mask[p] == 1:
                    state.particle_v[p] = velocity_modifier_params.velocity

        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)

    def restart_and_compute_F_C(self, mpm_model, mpm_state, target_pos, device):
        grid_size = (
            mpm_model.grid_dim_x,
            mpm_model.grid_dim_y,
            mpm_model.grid_dim_z,
        )

        wp.launch(
            kernel=zero_grid,  # gradient might gone
            dim=(grid_size),
            inputs=[mpm_state, mpm_model],
            device=device,
        )

        wp.launch(
            set_F_C_p2g,
            dim=self.n_particles,
            inputs=[mpm_state, mpm_model, target_pos],
            device=device,
        )

        wp.launch(
            kernel=grid_normalization_and_gravity,
            dim=(grid_size),
            inputs=[mpm_state, mpm_model, 0],
            device=device,
        )

        wp.launch(
            set_F_C_g2p,
            dim=self.n_particles,
            inputs=[mpm_state, mpm_model],
            device=device,
        )

        wp.launch(
            kernel=zero_grid,  # gradient might gone
            dim=(grid_size),
            inputs=[mpm_state, mpm_model],
            device=device,
        )

        # set position to target_pos
        wp.launch(
            kernel=set_vec3_to_vec3,
            dim=self.n_particles,
            inputs=[mpm_state.particle_x, target_pos],
            device=device,
        )