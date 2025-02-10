from typing import Optional, Tuple
from jaxtyping import Float, Int, Shaped
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor

import warp as wp

from thirdparty_code.warp_mpm.warp_utils import from_torch_safe, MyTape, CondTape
from thirdparty_code.warp_mpm.mpm_solver_diff import MPMWARPDiff
from thirdparty_code.warp_mpm.mpm_utils import compute_position_l2_loss, aggregate_grad, compute_posloss_with_grad
from thirdparty_code.warp_mpm.mpm_data_structure import MPMStateStruct, MPMModelStruct, get_float_array_product
from thirdparty_code.warp_mpm.mpm_utils import (compute_Closs_with_grad, compute_Floss_with_grad, 
                                                compute_posloss_with_grad, compute_veloloss_with_grad)


class MPMDifferentiableSimulation(autograd.Function):

    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        mpm_solver: MPMWARPDiff,
        mpm_state: MPMStateStruct,
        mpm_model: MPMModelStruct,
        substep: int, 
        substep_size: float, 
        num_substeps: int,
        init_pos: Float[Tensor, "n 3"],
        init_velocity: Float[Tensor, "n 3"],
        E: Float[Tensor, "n"] | Float[Tensor, "1"],
        nu: Float[Tensor, "n"] | Float[Tensor, "1"],
        particle_density: Optional[Float[Tensor, "n"] | Float[Tensor, "1"]]=None,
        density_change_mask: Optional[Int[Tensor, "n"]] = None,
        static_pos: Optional[Float[Tensor, "n 3"]] = None,
        device: str="cuda:0",
        requires_grad: bool=True,
        extra_no_grad_steps: int=0,
    ) -> Float[Tensor, "n 3"]:
        """
        Args:
            density_change_mask: [n] 0 or 1.  1 means the density of this particle can change.
        """
        
        num_particles = init_pos.shape[0]
        if static_pos is None:
            
            mpm_state.reset_state(
                init_pos.clone(),
                None,
                init_velocity, #.clone(),
                device=device,
                requires_grad=requires_grad,
            )
        else:
            mpm_state.reset_state(
                static_pos.clone(),
                None,
                init_velocity, #.clone(),
                device=device,
                requires_grad=requires_grad,

            )
            init_xyzs_wp = from_torch_safe(init_pos.clone().detach().contiguous(), dtype=wp.vec3, requires_grad=requires_grad)
            mpm_solver.restart_and_compute_F_C(mpm_model, mpm_state, init_xyzs_wp, device=device)
        
        if E.ndim == 0:
            E_inp = E.item() # float
            ctx.aggregating_E = True
        else:
            E_inp = from_torch_safe(E, dtype=wp.float32, requires_grad=requires_grad)
            ctx.aggregating_E = False
        if nu.ndim == 0:
            nu_inp = nu.item() # float
            ctx.aggregating_nu = True
        else:
            nu_inp = from_torch_safe(nu, dtype=wp.float32, requires_grad=requires_grad)
            ctx.aggregating_nu = False
            
        mpm_solver.set_E_nu(mpm_model, E_inp, nu_inp, device=device)

        mpm_state.reset_density(
            tensor_density=particle_density,
            selection_mask=density_change_mask,
            device=device,
            requires_grad=requires_grad)
        
        prev_state = mpm_state

        if extra_no_grad_steps > 0:
            with torch.no_grad():
                wp.launch(
                    kernel=get_float_array_product,
                    dim=num_particles,
                    inputs=[
                        mpm_state.particle_density,
                        mpm_state.particle_vol,
                        mpm_state.particle_mass,
                    ],
                    device=device,
                )
                mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device=device)

                for i in range(extra_no_grad_steps):
                    next_state = prev_state.partial_clone(requires_grad=requires_grad)
                    mpm_solver.p2g2p_differentiable(mpm_model, prev_state, next_state, substep_size, device=device)
                    prev_state = next_state
        else:
            prev_state = mpm_state

        wp_tape = MyTape()
        cond_tape: CondTape = CondTape(wp_tape, requires_grad)

        next_state_list = [] 
        
        with cond_tape:
            wp.launch(
                kernel=get_float_array_product,
                dim=num_particles,
                inputs=[
                    prev_state.particle_density,
                    prev_state.particle_vol,
                    prev_state.particle_mass,
                ],
                device=device,
            )
            mpm_solver.prepare_mu_lam(mpm_model, prev_state, device=device)

            for substep_local in range(num_substeps):
                next_state = prev_state.partial_clone(requires_grad=requires_grad)
                mpm_solver.p2g2p_differentiable(mpm_model, prev_state, next_state, substep_size, device=device)

                # next_state = mpm_solver.p2g2p_differentiable(mpm_model, prev_state, substep_size, device=device)
                next_state_list.append(next_state)
                prev_state = next_state
        
        ctx.mpm_solver = mpm_solver
        ctx.mpm_state = mpm_state
        ctx.mpm_model = mpm_model
        ctx.tape = cond_tape.tape
        ctx.device = device
        ctx.num_particles = num_particles

        ctx.next_state_list = next_state_list

        ctx.save_for_backward(density_change_mask)

        last_state = next_state_list[-1]
        particle_pos = wp.to_torch(last_state.particle_x).detach().clone()

        return particle_pos
    

    @staticmethod
    def backward(ctx, out_pos_grad: Float[Tensor, "n 3"]):
        
        num_particles = ctx.num_particles
        tape, device = ctx.tape, ctx.device
        mpm_solver, mpm_state, mpm_model = ctx.mpm_solver, ctx.mpm_state, ctx.mpm_model
        last_state = ctx.next_state_list[-1]
        density_change_mask = ctx.saved_tensors[0]

        grad_pos_wp = from_torch_safe(out_pos_grad, dtype=wp.vec3, requires_grad=False)
        target_pos_detach = wp.clone(last_state.particle_x, device=device, requires_grad=False)

        with tape:
            loss_wp = torch.zeros(1, device=device)
            loss_wp = wp.from_torch(loss_wp, requires_grad=True)
            wp.launch(
                compute_posloss_with_grad, 
                dim=num_particles,
                inputs=[
                    last_state,
                    target_pos_detach,
                    grad_pos_wp,
                    0.5,
                    loss_wp,
                ],
                device=device,
            )

        tape.backward(loss_wp)

        pos_grad = None
        if mpm_state.particle_v.grad is None:
            velo_grad = None
        else:
            velo_grad = wp.to_torch(mpm_state.particle_v.grad).detach().clone()

        # print("debug back", velo_grad)

        # grad for E, nu. TODO: add spatially varying E, nu later
        if ctx.aggregating_E:
            E_grad = wp.from_torch(torch.zeros(1, device=device), requires_grad=False)
            wp.launch(
                aggregate_grad,
                dim=num_particles,
                inputs=[
                    E_grad,
                    mpm_model.E.grad,
                ],
                device=device,
            )
            E_grad = wp.to_torch(E_grad)[0] / num_particles
        else:
            E_grad = wp.to_torch(mpm_model.E.grad).detach().clone()

        if ctx.aggregating_nu:
            nu_grad = wp.from_torch(torch.zeros(1, device=device), requires_grad=False)
            wp.launch(
                aggregate_grad,
                dim=num_particles,
                inputs=[nu_grad, mpm_model.nu.grad],
                device=device,
            )
            nu_grad = wp.to_torch(nu_grad)[0] / num_particles   
        else:
            nu_grad = wp.to_torch(mpm_model.nu.grad).detach().clone()

        # grad for density
        if mpm_state.particle_density.grad is None:
            density_grad = None
        else:
            density_grad = wp.to_torch(mpm_state.particle_density.grad).detach()
            density_grad = density_grad[density_change_mask.type(torch.bool)]
        
        density_mask_grad = None
        static_pos_grad = None 

        # from IPython import embed; embed()
        tape.zero()
        # print(density_grad.abs().sum(), velo_grad.abs().sum(), E_grad.abs().item(), nu_grad.abs().item(), "in sim func")

        return (None, None, None, None, None, None, 
                pos_grad, velo_grad, E_grad, nu_grad,
                 density_grad, density_mask_grad, 
                 static_pos_grad, None, None, None)
    


class MPMDifferentiableSimulationWCheckpoint(autograd.Function):
    """
    Current version does not support grad for density. 
    Please set vol, mass before calling this function.
    """

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx: autograd.function.FunctionCtx,
        mpm_solver: MPMWARPDiff,
        mpm_state: MPMStateStruct,
        mpm_model: MPMModelStruct,
        substep_size: float, 
        num_substeps: int,
        particle_x: Float[Tensor, "n 3"], 
        particle_v: Float[Tensor, "n 3"],
        particle_F: Float[Tensor, "n 3 3"],
        particle_C: Float[Tensor, "n 3 3"],
        E: Float[Tensor, "n"] | Float[Tensor, "1"],
        nu: Float[Tensor, "n"] | Float[Tensor, "1"],
        particle_density: Optional[Float[Tensor, "n"] | Float[Tensor, "1"]]=None,
        query_mask: Optional[Int[Tensor, "n"]] = None,
        device: str="cuda:0",
        requires_grad: bool=True,
        extra_no_grad_steps: int=0,
    ) -> Tuple[Float[Tensor, "n 3"], Float[Tensor, "n 3"], Float[Tensor, "n 9"], Float[Tensor, "n 9"]]:
        """
        Args:
            query_mask: [n] 0 or 1.  1 means the density or young's modulus, or poisson'ratio of this particle can change.
        """
        
        # initialization work is done before calling forward! 

        num_particles = particle_x.shape[0]

        mpm_state.continue_from_torch(
            particle_x, particle_v, particle_F, particle_C, device=device, requires_grad=True
        )
        # set x, v, F, C.

        if E.ndim == 0:
            E_inp = E.item() # float
            ctx.aggregating_E = True
        else:
            E_inp = from_torch_safe(E, dtype=wp.float32, requires_grad=True)
            ctx.aggregating_E = False
        if nu.ndim == 0:
            nu_inp = nu.item() # float
            ctx.aggregating_nu = True
        else:
            nu_inp = from_torch_safe(nu, dtype=wp.float32, requires_grad=True)
            ctx.aggregating_nu = False
            
        mpm_solver.set_E_nu(mpm_model, E_inp, nu_inp, device=device)
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device=device)

        mpm_state.reset_density(
            tensor_density=particle_density,
            selection_mask=query_mask,
            device=device,
            requires_grad=True,
            update_mass=True)
        
        prev_state = mpm_state

        if extra_no_grad_steps > 0:
            with torch.no_grad():
                for i in range(extra_no_grad_steps):
                    next_state = prev_state.partial_clone(requires_grad=True)
                    mpm_solver.p2g2p_differentiable(mpm_model, prev_state, next_state, substep_size, device=device)
                    prev_state = next_state

        # following steps will be checkpointed. then replayed in backward
        ctx.prev_state = prev_state
        
        for substep_local in range(num_substeps):
            next_state = prev_state.partial_clone(requires_grad=True)
            mpm_solver.p2g2p_differentiable(mpm_model, prev_state, next_state, substep_size, device=device)
            prev_state = next_state
        
        
        ctx.mpm_solver = mpm_solver
        ctx.mpm_state = mpm_state # state at the begining of this function; TODO: drop it?
        ctx.mpm_model = mpm_model
        ctx.device = device
        ctx.num_particles = num_particles

        ctx.num_substeps = num_substeps
        ctx.substep_size = substep_size
        
        ctx.save_for_backward(E, nu, particle_density, query_mask)

        last_state = next_state
        particle_pos = wp.to_torch(last_state.particle_x).detach().clone()
        particle_velo = wp.to_torch(last_state.particle_v).detach().clone()
        particle_F = wp.to_torch(last_state.particle_F_trial).detach().clone()
        particle_C = wp.to_torch(last_state.particle_C).detach().clone()

        return particle_pos, particle_velo, particle_F, particle_C
    

    @staticmethod
    def backward(ctx, out_pos_grad: Float[Tensor, "n 3"], out_velo_grad: Float[Tensor, "n 3"], 
                 out_F_grad: Float[Tensor, "n 9"], out_C_grad: Float[Tensor, "n 9"]):
        
        num_particles = ctx.num_particles
        device = ctx.device
        mpm_solver, mpm_model = ctx.mpm_solver, ctx.mpm_model
        prev_state = ctx.prev_state
        starting_state = ctx.prev_state 

        E, nu, particle_density, query_mask = ctx.saved_tensors

        num_substeps, substep_size = ctx.num_substeps, ctx.substep_size

        # rolling back
        # setting initial param first: 
        if E.ndim == 0:
            E_inp = E.item() # float
            ctx.aggregating_E = True
        else:
            E_inp = from_torch_safe(E, dtype=wp.float32, requires_grad=True)
            ctx.aggregating_E = False
        if nu.ndim == 0:
            nu_inp = nu.item() # float
            ctx.aggregating_nu = True
        else:
            nu_inp = from_torch_safe(nu, dtype=wp.float32, requires_grad=True)
            ctx.aggregating_nu = False
            
        mpm_solver.set_E_nu(mpm_model, E_inp, nu_inp, device=device)

        starting_state.reset_density(
            tensor_density=particle_density,
            selection_mask=query_mask,
            device=device,
            requires_grad=True)
        
        next_state_list = []

        with wp.ScopedDevice(device):
            tape = MyTape()

            # handle it later
            grad_pos_wp = from_torch_safe(out_pos_grad, dtype=wp.vec3, requires_grad=False)
            if out_velo_grad is not None:
                grad_velo_wp = from_torch_safe(out_velo_grad, dtype=wp.vec3, requires_grad=False)
            else:
                grad_velo_wp = None
            
            if out_F_grad is not None:
                grad_F_wp = from_torch_safe(out_F_grad, dtype=wp.mat33, requires_grad=False)
            else:
                grad_F_wp = None
            
            if out_C_grad is not None:
                grad_C_wp = from_torch_safe(out_C_grad, dtype=wp.mat33, requires_grad=False)
            else:
                grad_C_wp = None

            with tape:

                wp.launch(
                    kernel=get_float_array_product,
                    dim=num_particles,
                    inputs=[
                        prev_state.particle_density,
                        prev_state.particle_vol,
                        prev_state.particle_mass,
                    ],
                    device=device,
                )
                mpm_solver.prepare_mu_lam(mpm_model, prev_state, device=device)

                for substep_local in range(num_substeps):
                    next_state = prev_state.partial_clone(requires_grad=True)
                    mpm_solver.p2g2p_differentiable(mpm_model, prev_state, next_state, substep_size, device=device)

                    # next_state = mpm_solver.p2g2p_differentiable(mpm_model, prev_state, substep_size, device=device)
                    next_state_list.append(next_state)
                    prev_state = next_state

                # simulation done. Compute loss:
                
                loss_wp = torch.zeros(1, device=device)
                loss_wp = wp.from_torch(loss_wp, requires_grad=True)
                target_pos_detach = wp.clone(next_state.particle_x, device=device, requires_grad=False)
                wp.launch(
                    compute_posloss_with_grad, 
                    dim=num_particles,
                    inputs=[
                        next_state,
                        target_pos_detach,
                        grad_pos_wp,
                        0.5,
                        loss_wp,
                    ],
                    device=device,
                )
                if grad_velo_wp is not None:
                    target_velo_detach = wp.clone(next_state.particle_v, device=device, requires_grad=False)
                    wp.launch(
                        compute_veloloss_with_grad, 
                        dim=num_particles,
                        inputs=[
                            next_state,
                            target_velo_detach,
                            grad_velo_wp,
                            0.5,
                            loss_wp,
                        ],
                        device=device,
                    )
                
                if grad_F_wp is not None:
                    target_F_detach = wp.clone(next_state.particle_F_trial, device=device, requires_grad=False)
                    wp.launch(
                        compute_Floss_with_grad, 
                        dim=num_particles,
                        inputs=[
                            next_state,
                            target_F_detach,
                            grad_F_wp,
                            0.5,
                            loss_wp,
                        ],
                        device=device,
                    )
                if grad_C_wp is not None:
                    target_C_detach = wp.clone(next_state.particle_C, device=device, requires_grad=False)
                    wp.launch(
                        compute_Closs_with_grad, 
                        dim=num_particles,
                        inputs=[
                            next_state,
                            target_C_detach,
                            grad_C_wp,
                            0.5,
                            loss_wp,
                        ],
                        device=device,)

            # wp.synchronize_device(device)            
            tape.backward(loss_wp)
            # from IPython import embed; embed()

        pos_grad = wp.to_torch(starting_state.particle_x.grad).detach().clone()
        velo_grad = wp.to_torch(starting_state.particle_v.grad).detach().clone()
        F_grad = wp.to_torch(starting_state.particle_F_trial.grad).detach().clone()
        C_grad = wp.to_torch(starting_state.particle_C.grad).detach().clone()
        # print("debug back", velo_grad)

        # grad for E, nu. TODO: add spatially varying E, nu later
        if ctx.aggregating_E:
            E_grad = wp.from_torch(torch.zeros(1, device=device), requires_grad=False)
            wp.launch(
                aggregate_grad,
                dim=num_particles,
                inputs=[
                    E_grad,
                    mpm_model.E.grad,
                ],
                device=device,
            )
            E_grad = wp.to_torch(E_grad)[0] / num_particles
        else:
            E_grad = wp.to_torch(mpm_model.E.grad).detach().clone()

        if ctx.aggregating_nu:
            nu_grad = wp.from_torch(torch.zeros(1, device=device), requires_grad=False)
            wp.launch(
                aggregate_grad,
                dim=num_particles,
                inputs=[nu_grad, mpm_model.nu.grad],
                device=device,
            )
            nu_grad = wp.to_torch(nu_grad)[0] / num_particles   
        else:
            nu_grad = wp.to_torch(mpm_model.nu.grad).detach().clone()

        # grad for density
        if starting_state.particle_density.grad is None:
            density_grad = None
        else:
            density_grad = wp.to_torch(starting_state.particle_density.grad).detach()

        
        density_mask_grad = None
        static_pos_grad = None 

        tape.zero()
        # print(density_grad.abs().sum(), velo_grad.abs().sum(), E_grad.abs().item(), nu_grad.abs().item(), "in sim func")
        # from IPython import embed; embed()
        
        return (None, None, None, None, None,
                pos_grad, velo_grad, F_grad, C_grad, 
                E_grad, nu_grad,
                density_grad, density_mask_grad, 
                None, None, None)


class MPMDifferentiableSimulationClean(autograd.Function):
    """
    Current version does not support grad for density. 
    Please set vol, mass before calling this function.
    """

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx: autograd.function.FunctionCtx,
        mpm_solver: MPMWARPDiff,
        mpm_state: MPMStateStruct,
        mpm_model: MPMModelStruct,
        substep_size: float, 
        num_substeps: int,
        particle_x: Float[Tensor, "n 3"], 
        particle_v: Float[Tensor, "n 3"],
        particle_F: Float[Tensor, "n 3 3"],
        particle_C: Float[Tensor, "n 3 3"],
        E: Float[Tensor, "n"] | Float[Tensor, "1"],
        nu: Float[Tensor, "n"] | Float[Tensor, "1"],
        particle_density: Optional[Float[Tensor, "n"] | Float[Tensor, "1"]]=None,
        query_mask: Optional[Int[Tensor, "n"]] = None,
        device: str="cuda:0",
        requires_grad: bool=True,
        extra_no_grad_steps: int=0,
    ) -> Tuple[Float[Tensor, "n 3"], Float[Tensor, "n 3"], Float[Tensor, "n 9"], Float[Tensor, "n 9"], Float[Tensor, "n 6"]]:
        """
        Args:
            query_mask: [n] 0 or 1.  1 means the density or young's modulus, or poisson'ratio of this particle can change.
        """
        
        # initialization work is done before calling forward! 

        num_particles = particle_x.shape[0]

        mpm_state.continue_from_torch(
            particle_x, particle_v, particle_F, particle_C, device=device, requires_grad=True
        )
        # set x, v, F, C.

        if E.ndim == 0:
            E_inp = E.item() # float
            ctx.aggregating_E = True
        else:
            E_inp = from_torch_safe(E, dtype=wp.float32, requires_grad=True)
            ctx.aggregating_E = False
        if nu.ndim == 0:
            nu_inp = nu.item() # float
            ctx.aggregating_nu = True
        else:
            nu_inp = from_torch_safe(nu, dtype=wp.float32, requires_grad=True)
            ctx.aggregating_nu = False
            
        mpm_solver.set_E_nu(mpm_model, E_inp, nu_inp, device=device)
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device=device)

        mpm_state.reset_density(
            tensor_density=particle_density,
            selection_mask=query_mask,
            device=device,
            requires_grad=True,
            update_mass=True)
        
        prev_state = mpm_state

        if extra_no_grad_steps > 0:
            with torch.no_grad():
                for i in range(extra_no_grad_steps):
                    next_state = prev_state.partial_clone(requires_grad=True)
                    mpm_solver.p2g2p_differentiable(mpm_model, prev_state, next_state, substep_size, device=device)
                    prev_state = next_state

        # following steps will be checkpointed. then replayed in backward
        ctx.prev_state = prev_state

        wp_tape = MyTape()
        cond_tape: CondTape = CondTape(wp_tape, requires_grad)
        next_state_list = [] 

        with cond_tape:
            wp.launch(
                kernel=get_float_array_product,
                dim=num_particles,
                inputs=[
                    prev_state.particle_density,
                    prev_state.particle_vol,
                    prev_state.particle_mass,
                ],
                device=device,
            )
            mpm_solver.prepare_mu_lam(mpm_model, prev_state, device=device)

            for substep_local in range(num_substeps):
                next_state = prev_state.partial_clone(requires_grad=True)
                mpm_solver.p2g2p_differentiable(mpm_model, prev_state, next_state, substep_size, device=device)
                next_state_list.append(next_state)
                prev_state = next_state
        
        ctx.mpm_solver = mpm_solver
        ctx.mpm_model = mpm_model
        ctx.next_state_list = next_state_list
        ctx.device = device
        ctx.num_particles = num_particles
        ctx.tape = cond_tape.tape

        ctx.save_for_backward(query_mask)

        last_state = next_state
        particle_pos = wp.to_torch(last_state.particle_x).detach().clone()
        particle_velo = wp.to_torch(last_state.particle_v).detach().clone()
        particle_F = wp.to_torch(last_state.particle_F_trial).detach().clone()
        particle_C = wp.to_torch(last_state.particle_C).detach().clone()
        # [N * 6, ]
        particle_cov = wp.to_torch(last_state.particle_cov).detach().clone()

        particle_cov = particle_cov.view(-1, 6)

        return particle_pos, particle_velo, particle_F, particle_C, particle_cov
    

    @staticmethod
    def backward(ctx, out_pos_grad: Float[Tensor, "n 3"], out_velo_grad: Float[Tensor, "n 3"], 
                 out_F_grad: Float[Tensor, "n 9"], out_C_grad: Float[Tensor, "n 9"], out_cov_grad: Float[Tensor, "n 6"]):
        
        num_particles = ctx.num_particles
        device = ctx.device
        mpm_solver, mpm_model = ctx.mpm_solver, ctx.mpm_model
        tape = ctx.tape
        starting_state = ctx.prev_state
        
        next_state_list = ctx.next_state_list
        next_state = next_state_list[-1]

        query_mask = ctx.saved_tensors
    
        with wp.ScopedDevice(device):
            
            grad_pos_wp = from_torch_safe(out_pos_grad, dtype=wp.vec3, requires_grad=False)
            
            with tape:
                loss_wp = torch.zeros(1, device=device)
                loss_wp = wp.from_torch(loss_wp, requires_grad=True)
                target_pos_detach = wp.clone(next_state.particle_x, device=device, requires_grad=False)
                wp.launch(
                    compute_posloss_with_grad, 
                    dim=num_particles,
                    inputs=[
                        next_state,
                        target_pos_detach,
                        grad_pos_wp,
                        0.5,
                        loss_wp,
                    ],
                    device=device,
                )

            # wp.synchronize_device(device)            
            tape.backward(loss_wp)
            # from IPython import embed; embed()

        pos_grad = wp.to_torch(starting_state.particle_x.grad).detach().clone()
        velo_grad = wp.to_torch(starting_state.particle_v.grad).detach().clone()
        F_grad = wp.to_torch(starting_state.particle_F_trial.grad).detach().clone()
        C_grad = wp.to_torch(starting_state.particle_C.grad).detach().clone()
        # print("debug back", velo_grad)

        # grad for E, nu. TODO: add spatially varying E, nu later
        if ctx.aggregating_E:
            E_grad = wp.from_torch(torch.zeros(1, device=device), requires_grad=False)
            wp.launch(
                aggregate_grad,
                dim=num_particles,
                inputs=[
                    E_grad,
                    mpm_model.E.grad,
                ],
                device=device,
            )
            E_grad = wp.to_torch(E_grad)[0] / num_particles
        else:
            E_grad = wp.to_torch(mpm_model.E.grad).detach().clone()

        if ctx.aggregating_nu:
            nu_grad = wp.from_torch(torch.zeros(1, device=device), requires_grad=False)
            wp.launch(
                aggregate_grad,
                dim=num_particles,
                inputs=[nu_grad, mpm_model.nu.grad],
                device=device,
            )
            nu_grad = wp.to_torch(nu_grad)[0] / num_particles   
        else:
            nu_grad = wp.to_torch(mpm_model.nu.grad).detach().clone()

        # grad for density
        if starting_state.particle_density.grad is None:
            density_grad = None
        else:
            density_grad = wp.to_torch(starting_state.particle_density.grad).detach()
        density_mask_grad = None

        tape.zero()
        # print(density_grad.abs().sum(), velo_grad.abs().sum(), E_grad.abs().item(), nu_grad.abs().item(), "in sim func")
        # from IPython import embed; embed()
        
        return (None, None, None, None, None,
                pos_grad, velo_grad, F_grad, C_grad, 
                E_grad, nu_grad,
                density_grad, density_mask_grad, 
                None, None, None)