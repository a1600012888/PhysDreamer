import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from torch import Tensor
from jaxtyping import Float, Int, Shaped
from typing import List

import point_cloud_utils as pcu

from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

import numpy as np
import logging
import argparse
import shutil
import wandb
import torch
import os
from motionrep.utils.config import create_config
from motionrep.utils.optimizer import get_linear_schedule_with_warmup
from time import time
from omegaconf import OmegaConf
import numpy as np

# from motionrep.utils.torch_utils import get_sync_time
from einops import rearrange, repeat

from motionrep.gaussian_3d.gaussian_renderer.feat_render import render_feat_gaussian
from motionrep.gaussian_3d.scene import GaussianModel

from motionrep.data.datasets.multiview_dataset import MultiviewImageDataset
from motionrep.data.datasets.multiview_video_dataset import (
    MultiviewVideoDataset,
    camera_dataset_collate_fn,
)

from motionrep.data.datasets.multiview_dataset import (
    camera_dataset_collate_fn as camera_dataset_collate_fn_img,
)

from typing import NamedTuple
import torch.nn.functional as F

from motionrep.utils.img_utils import compute_psnr, compute_ssim
from thirdparty_code.warp_mpm.mpm_data_structure import (
    MPMStateStruct,
    MPMModelStruct,
    get_float_array_product,
)
from thirdparty_code.warp_mpm.mpm_solver_diff import MPMWARPDiff
from thirdparty_code.warp_mpm.warp_utils import from_torch_safe
from thirdparty_code.warp_mpm.gaussian_sim_utils import get_volume
import warp as wp
import random

from local_utils import (
    cycle,
    create_spatial_fields,
    find_far_points,
    LinearStepAnneal,
    apply_grid_bc_w_freeze_pts,
    render_gaussian_seq_w_mask_with_disp,
    downsample_with_kmeans_gpu,
)
from interface import MPMDifferentiableSimulation

logger = get_logger(__name__, log_level="INFO")


def create_dataset(args):
    assert args.dataset_res in ["middle", "small", "large"]
    if args.dataset_res == "middle":
        res = [320, 576]
    elif args.dataset_res == "small":
        res = [192, 320]
    elif args.dataset_res == "large":
        res = [576, 1024]
    else:
        raise NotImplementedError

    video_dir_name = "videos_2"
    dataset = MultiviewVideoDataset(
        args.dataset_dir,
        use_white_background=False,
        resolution=res,
        scale_x_angle=1.0,
        video_dir_name=video_dir_name,
    )

    test_dataset = MultiviewImageDataset(
        args.dataset_dir,
        use_white_background=False,
        resolution=res,
        # use_index=list(range(0, 30, 4)),
        # use_index=[0],
        scale_x_angle=1.0,
        fitler_with_renderd=True,
        load_imgs=False,
    )
    print("len of test dataset", len(test_dataset))
    return dataset, test_dataset


class Trainer:
    def __init__(self, args):
        self.args = args

        self.ssim = args.ssim
        args.warmup_step = int(args.warmup_step * args.gradient_accumulation_steps)
        args.train_iters = int(args.train_iters * args.gradient_accumulation_steps)
        os.environ["WANDB__SERVICE_WAIT"] = "600"
        args.wandb_name += (
            "decay_{}_substep_{}_{}_lr_{}_tv_{}_iters_{}_sw_{}_cw_{}".format(
                args.loss_decay,
                args.substep,
                args.model,
                args.lr,
                args.tv_loss_weight,
                args.train_iters,
                args.start_window_size,
                args.compute_window,
            )
        )

        logging_dir = os.path.join(args.output_dir, args.wandb_name)
        accelerator_project_config = ProjectConfiguration(logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=1,  # args.gradient_accumulation_steps,
            mixed_precision="no",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
        )
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        set_seed(args.seed + accelerator.process_index)
        print("process index", accelerator.process_index)
        if accelerator.is_main_process:
            output_path = os.path.join(logging_dir, f"seed{args.seed}")
            os.makedirs(output_path, exist_ok=True)
            self.output_path = output_path

        self.rand_bg = args.rand_bg
        # setup the dataset
        dataset, test_dataset = create_dataset(args)
        self.test_dataset = test_dataset

        dataset_dir = test_dataset.data_dir
        self.dataset = dataset

        gaussian_path = os.path.join(dataset_dir, "point_cloud.ply")
        aabb = self.setup_eval(
            args,
            gaussian_path,
            white_background=True,
        )
        self.aabb = aabb

        self.num_frames = int(args.num_frames)
        self.window_size_schduler = LinearStepAnneal(
            args.train_iters,
            start_state=[args.start_window_size],
            end_state=[13],
            plateau_iters=0,
            warmup_step=300,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            collate_fn=camera_dataset_collate_fn_img,
        )
        # why prepare here again?
        test_dataloader = accelerator.prepare(test_dataloader)
        self.test_dataloader = cycle(test_dataloader)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=camera_dataset_collate_fn,
        )
        # why prepare here again?
        dataloader = accelerator.prepare(dataloader)
        self.dataloader = cycle(dataloader)

        self.train_iters = args.train_iters
        self.accelerator = accelerator
        # init traiable params
        E_nu_list = self.init_trainable_params()
        for p in E_nu_list:
            p.requires_grad = True
        self.E_nu_list = E_nu_list

        self.setup_simulation(dataset_dir, grid_size=args.grid_size)

        if args.checkpoint_path == "None":
            args.checkpoint_path = None
        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)
            trainable_params = list(self.sim_fields.parameters()) + self.E_nu_list
            optim_list = [
                {"params": self.E_nu_list, "lr": args.lr * 1e-10},
                {
                    "params": self.sim_fields.parameters(),
                    "lr": args.lr,
                    "weight_decay": 1e-4,
                },
                # {"params": self.velo_fields.parameters(), "lr": args.lr * 1e-3, "weight_decay": 1e-4},
            ]
            self.freeze_velo = True
            self.velo_optimizer = None
        else:
            trainable_params = list(self.sim_fields.parameters()) + self.E_nu_list
            optim_list = [
                {"params": self.E_nu_list, "lr": args.lr * 1e-10},
                {
                    "params": self.sim_fields.parameters(),
                    "lr": args.lr * 1e-10,
                    "weight_decay": 1e-4,
                },
            ]
            self.freeze_velo = False
            self.window_size_schduler.warmup_step = 800

            velo_optim = [
                {
                    "params": self.velo_fields.parameters(),
                    "lr": args.lr * 0.1,
                    "weight_decay": 1e-4,
                },
            ]
            self.velo_optimizer = torch.optim.AdamW(
                velo_optim,
                lr=args.lr,
                weight_decay=0.0,
            )
            self.velo_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.velo_optimizer,
                num_warmup_steps=args.warmup_step,
                num_training_steps=args.train_iters,
            )
            self.velo_optimizer, self.velo_scheduler = accelerator.prepare(
                self.velo_optimizer, self.velo_scheduler
            )

        self.optimizer = torch.optim.AdamW(
            optim_list,
            lr=args.lr,
            weight_decay=0.0,
        )
        self.trainable_params = trainable_params
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters,
        )
        self.sim_fields, self.optimizer, self.scheduler = accelerator.prepare(
            self.sim_fields, self.optimizer, self.scheduler
        )
        self.velo_fields = accelerator.prepare(self.velo_fields)

        # setup train info
        self.step = 0
        self.batch_size = args.batch_size
        self.tv_loss_weight = args.tv_loss_weight

        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.max_grad_norm = args.max_grad_norm

        self.use_wandb = args.use_wandb
        if self.accelerator.is_main_process:
            if args.use_wandb:
                run = wandb.init(
                    config=dict(args),
                    dir=self.output_path,
                    **{
                        "mode": "online",
                        "entity": args.wandb_entity,
                        "project": args.wandb_project,
                    },
                )
                wandb.run.log_code(".")
                wandb.run.name = args.wandb_name
                print(f"run dir: {run.dir}")
                self.wandb_folder = run.dir
                os.makedirs(self.wandb_folder, exist_ok=True)

    def init_trainable_params(
        self,
    ):

        # init young modulus and poisson ratio

        young_numpy = np.exp(np.random.uniform(np.log(1e-3), np.log(1e3))).astype(
            np.float32
        )
        young_numpy = 1e6 * 1.0

        young_modulus = torch.tensor(young_numpy, dtype=torch.float32).to(
            self.accelerator.device
        )

        poisson_numpy = np.random.uniform(0.1, 0.4)
        poisson_ratio = torch.tensor(poisson_numpy, dtype=torch.float32).to(
            self.accelerator.device
        )

        trainable_params = [young_modulus, poisson_ratio]

        print(
            "init young modulus: ",
            young_modulus.item(),
            "poisson ratio: ",
            poisson_ratio.item(),
        )
        return trainable_params

    def setup_simulation(self, dataset_dir, grid_size=100):

        device = "cuda:{}".format(self.accelerator.process_index)

        xyzs = self.render_params.gaussians.get_xyz.detach().clone()
        sim_xyzs = xyzs[self.sim_mask_in_raw_gaussian, :]
        sim_cov = (
            self.render_params.gaussians.get_covariance()[
                self.sim_mask_in_raw_gaussian, :
            ]
            .detach()
            .clone()
        )

        # scale, and shift
        pos_max = sim_xyzs.max()
        pos_min = sim_xyzs.min()
        scale = (pos_max - pos_min) * 1.8
        shift = -pos_min + (pos_max - pos_min) * 0.25
        self.scale, self.shift = scale, shift
        print("scale, shift", scale, shift)

        # filled
        filled_in_points_path = os.path.join(dataset_dir, "internal_filled_points.ply")

        if os.path.exists(filled_in_points_path):
            fill_xyzs = pcu.load_mesh_v(filled_in_points_path)  # [n, 3]
            fill_xyzs = fill_xyzs[
                np.random.choice(
                    fill_xyzs.shape[0], int(fill_xyzs.shape[0] * 0.25), replace=False
                )
            ]
            fill_xyzs = torch.from_numpy(fill_xyzs).float().to("cuda")
            self.fill_xyzs = fill_xyzs
            print(
                "loaded {} internal filled points from: ".format(fill_xyzs.shape[0]),
                filled_in_points_path,
            )
        else:
            self.fill_xyzs = None

        if self.fill_xyzs is not None:
            render_mask_in_sim_pts = torch.cat(
                [
                    torch.ones_like(sim_xyzs[:, 0]).bool(),
                    torch.zeros_like(fill_xyzs[:, 0]).bool(),
                ],
                dim=0,
            ).to(device)
            sim_xyzs = torch.cat([sim_xyzs, fill_xyzs], dim=0)
            sim_cov = torch.cat(
                [sim_cov, sim_cov.new_ones((fill_xyzs.shape[0], sim_cov.shape[-1]))],
                dim=0,
            )
            self.render_mask = render_mask_in_sim_pts
        else:
            self.render_mask = torch.ones_like(sim_xyzs[:, 0]).bool().to(device)

        sim_xyzs = (sim_xyzs + shift) / scale

        sim_aabb = torch.stack(
            [torch.min(sim_xyzs, dim=0)[0], torch.max(sim_xyzs, dim=0)[0]], dim=0
        )
        sim_aabb = (
            sim_aabb - torch.mean(sim_aabb, dim=0, keepdim=True)
        ) * 1.2 + torch.mean(sim_aabb, dim=0, keepdim=True)

        print("simulation aabb: ", sim_aabb)

        # point cloud resample with kmeans
        downsample_scale = self.args.downsample_scale
        num_cluster = int(sim_xyzs.shape[0] * downsample_scale)
        sim_xyzs = downsample_with_kmeans_gpu(sim_xyzs, num_cluster)

        sim_gaussian_pos = self.render_params.gaussians.get_xyz.detach().clone()[
            self.sim_mask_in_raw_gaussian, :
        ]
        sim_gaussian_pos = (sim_gaussian_pos + shift) / scale

        cdist = torch.cdist(sim_gaussian_pos, sim_xyzs) * -1.0
        _, top_k_index = torch.topk(cdist, self.args.top_k, dim=-1)
        self.top_k_index = top_k_index

        print("Downsampled to: ", sim_xyzs.shape[0], "by", downsample_scale)

        # compute volue for each point.
        points_volume = get_volume(sim_xyzs.detach().cpu().numpy())

        num_particles = sim_xyzs.shape[0]

        wp.init()
        wp.config.mode = "debug"
        wp.config.verify_cuda = True

        mpm_state = MPMStateStruct()
        mpm_state.init(num_particles, device=device, requires_grad=True)

        self.particle_init_position = sim_xyzs.clone()

        mpm_state.from_torch(
            self.particle_init_position.clone(),
            torch.from_numpy(points_volume).float().to(device).clone(),
            None,  # set cov to None, since it is not used.
            device=device,
            requires_grad=True,
            n_grid=grid_size,
            grid_lim=1.0,
        )
        mpm_model = MPMModelStruct()
        mpm_model.init(num_particles, device=device, requires_grad=True)
        mpm_model.init_other_params(n_grid=grid_size, grid_lim=1.0, device=device)

        material_params = {
            "material": "jelly",  # "jelly", "metal", "sand", "foam", "snow", "plasticine", "neo-hookean"
            "g": [0.0, 0.0, 0.0],
            "density": 2000,  # kg / m^3
            "grid_v_damping_scale": 1.1,  # 0.999,
        }

        self.v_damping = material_params["grid_v_damping_scale"]
        self.material_name = material_params["material"]
        mpm_solver = MPMWARPDiff(
            num_particles, n_grid=grid_size, grid_lim=1.0, device=device
        )
        mpm_solver.set_parameters_dict(mpm_model, mpm_state, material_params)

        self.mpm_state, self.mpm_model, self.mpm_solver = (
            mpm_state,
            mpm_model,
            mpm_solver,
        )

        # setup boundary condition:
        moving_pts_path = os.path.join(dataset_dir, "moving_part_points.ply")
        if os.path.exists(moving_pts_path):
            moving_pts = pcu.load_mesh_v(moving_pts_path)
            moving_pts = torch.from_numpy(moving_pts).float().to(device)
            moving_pts = (moving_pts + shift) / scale
            freeze_mask = find_far_points(
                sim_xyzs, moving_pts, thres=0.25 / grid_size
            ).bool()
            freeze_pts = sim_xyzs[freeze_mask, :]

            grid_freeze_mask = apply_grid_bc_w_freeze_pts(
                grid_size, 1.0, freeze_pts, mpm_solver
            )
            self.freeze_mask = freeze_mask

            # does not prefer boundary condition on particle
            # freeze_mask_select = setup_boundary_condition_with_points(sim_xyzs, moving_pts,
            #                                                         self.mpm_solver, self.mpm_state, thres=0.5 / grid_size)
            # self.freeze_mask = freeze_mask_select.bool()
        else:
            raise NotImplementedError

        num_freeze_pts = self.freeze_mask.sum()
        print(
            "num freeze pts in total",
            num_freeze_pts.item(),
            "num moving pts",
            num_particles - num_freeze_pts.item(),
        )

        # init fields for simulation, e.g. density, external force, etc.

        # padd init density, youngs,
        density = (
            torch.ones_like(self.particle_init_position[..., 0])
            * material_params["density"]
        )
        youngs_modulus = (
            torch.ones_like(self.particle_init_position[..., 0])
            * self.E_nu_list[0].detach()
        )
        poisson_ratio = torch.ones_like(self.particle_init_position[..., 0]) * 0.3

        # load stem for higher density
        stem_pts_path = os.path.join(dataset_dir, "stem_points.ply")
        if os.path.exists(stem_pts_path):
            stem_pts = pcu.load_mesh_v(stem_pts_path)
            stem_pts = torch.from_numpy(stem_pts).float().to(device)
            stem_pts = (stem_pts + shift) / scale
            no_stem_mask = find_far_points(
                sim_xyzs, stem_pts, thres=2.0 / grid_size
            ).bool()
            stem_mask = torch.logical_not(no_stem_mask)
            density[stem_mask] = 2000
            print("num stem pts", stem_mask.sum().item())

        self.density = density
        self.young_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio

        # set density, youngs, poisson
        mpm_state.reset_density(
            density.clone(),
            torch.ones_like(density).type(torch.int),
            device,
            update_mass=True,
        )
        mpm_solver.set_E_nu_from_torch(
            mpm_model, youngs_modulus.clone(), poisson_ratio.clone(), device
        )
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device)

        self.sim_fields = create_spatial_fields(self.args, 1, sim_aabb)
        self.sim_fields.train()

        self.args.sim_res = 24
        # self.velo_fields = create_velocity_model(self.args, sim_aabb)
        self.velo_fields = create_spatial_fields(
            self.args, 3, sim_aabb, add_entropy=False
        )
        self.velo_fields.train()

    def set_simulation_state(
        self,
        init_xyzs,
        init_velocity,
        device,
        requires_grad=False,
        use_precompute_F=False,
        use_density=True,
    ):

        initial_position_time0 = self.particle_init_position.clone()

        if use_precompute_F:
            self.mpm_state.reset_state(
                initial_position_time0,
                None,
                init_velocity.clone(),
                device=device,
                requires_grad=True,
            )

            init_xyzs_wp = from_torch_safe(
                init_xyzs.clone().detach().contiguous(),
                dtype=wp.vec3,
                requires_grad=True,
            )
            self.mpm_solver.restart_and_compute_F_C(
                self.mpm_model, self.mpm_state, init_xyzs_wp, device=device
            )
        else:
            self.mpm_state.reset_state(
                init_xyzs.clone(),
                None,
                init_velocity.clone(),
                device=device,
                requires_grad=True,
            )

    def get_density_velocity(self, time_stamp: float, device, requires_grad=True):

        initial_position_time0 = self.particle_init_position.clone()

        query_mask = torch.logical_not(self.freeze_mask)
        query_pts = initial_position_time0[query_mask, :]
        sim_params = self.sim_fields(query_pts)
        # density = sim_params[..., 0]

        # 0.1
        young_modulus = sim_params[..., 0]
        # young_modulus = torch.exp(sim_params[..., 0]) + init_young
        young_modulus = torch.clamp(young_modulus, 1e-3, 1e8)

        # young_padded = torch.ones_like(initial_position_time0[..., 0]) * init_young
        young_padded = self.young_modulus.detach().clone()
        young_padded[query_mask] = young_padded[query_mask] + young_modulus * 1

        density = self.density.detach().clone()

        velocity = self.velo_fields(query_pts)[..., :3]

        # scaling.
        velocity = velocity * 0.1

        return density, young_padded, velocity, query_mask

    def train_one_step(self):

        self.sim_fields.train()
        self.velo_fields.train()
        accelerator = self.accelerator
        device = "cuda:{}".format(accelerator.process_index)
        data = next(self.dataloader)
        cam = data["cam"][0]

        time_stamps = np.linspace(0, 1, self.num_frames).astype(np.float32)[1:]

        gt_videos = data["video_clip"][0, 1 : self.num_frames, ...]

        window_size = int(self.window_size_schduler.compute_state(self.step)[0])
        print("window size", window_size)
        stop_velo_opt_thres = 4
        do_velo_opt = not self.freeze_velo
        if not do_velo_opt:
            stop_velo_opt_thres = (
                0  # stop velocity optimization if we are loading from checkpoint
            )
        if window_size >= stop_velo_opt_thres:
            self.velo_fields.eval()
            do_velo_opt = False

        rendered_video_list = []
        log_loss_dict = {
            "loss": [],
            "l2_loss": [],
            "psnr": [],
            "ssim": [],
        }

        init_xyzs = self.particle_init_position.clone()
        num_particles = init_xyzs.shape[0]
        # delta_time = 1.0 / (self.num_frames - 1)
        delta_time = 1.0 / 30
        substep_size = delta_time / self.args.substep
        num_substeps = int(delta_time / substep_size)

        start_time_idx = max(0, window_size - self.args.compute_window)
        for time_idx in range(start_time_idx, window_size):
            # time_stamp = time_stamps[time_idx]
            time_stamp = time_stamps[0]  # fix to begining.. Start at the begining

            density, youngs_padded, init_velocity, query_mask = (
                self.get_density_velocity(time_stamp, device)
            )

            if not do_velo_opt:
                init_velocity = init_velocity.detach()
            padded_velocity = torch.zeros_like(init_xyzs)
            padded_velocity[query_mask, :] = init_velocity

            gt_frame = gt_videos[[time_idx]]

            extra_no_grad_step = max(
                0, (time_idx - self.args.grad_window + 1) * num_substeps
            )
            if do_velo_opt:
                extra_no_grad_step = 0

            num_step_with_grad = num_substeps * (time_idx + 1) - extra_no_grad_step

            particle_pos = MPMDifferentiableSimulation.apply(
                self.mpm_solver,
                self.mpm_state,
                self.mpm_model,
                0,
                substep_size,
                num_step_with_grad,
                init_xyzs,
                padded_velocity,
                youngs_padded,
                self.E_nu_list[1],
                density,
                query_mask,
                None,
                device,
                True,
                extra_no_grad_step,
            )

            gaussian_pos = particle_pos * self.scale - self.shift
            undeformed_gaussian_pos = (
                self.particle_init_position * self.scale - self.shift
            )
            disp_offset = gaussian_pos - undeformed_gaussian_pos.detach()
            # gaussian_pos.requires_grad = True

            simulated_video = render_gaussian_seq_w_mask_with_disp(
                cam,
                self.render_params,
                undeformed_gaussian_pos.detach(),
                self.top_k_index,
                [disp_offset],
                self.sim_mask_in_raw_gaussian,
            )

            # print("debug", simulated_video.shape, gt_frame.shape, gaussian_pos.shape, init_xyzs.shape, density.shape, query_mask.sum().item())
            rendered_video_list.append(simulated_video.detach())

            l2_loss = 0.5 * F.mse_loss(simulated_video, gt_frame, reduction="mean")
            ssim_loss = compute_ssim(simulated_video, gt_frame)
            loss = l2_loss * (1.0 - self.ssim) + (1.0 - ssim_loss) * self.ssim

            sm_velo_loss = self.velo_fields.compute_smoothess_loss()
            if time_idx > 2 or window_size > stop_velo_opt_thres:
                sm_velo_loss = sm_velo_loss.detach()
            sm_spatial_loss = self.sim_fields.compute_smoothess_loss()

            sm_loss = sm_velo_loss + sm_spatial_loss
            loss = (
                loss * (self.args.loss_decay**time_idx) + sm_loss * self.tv_loss_weight
            )
            loss = loss / self.args.compute_window
            loss.backward()

            with torch.no_grad():
                psnr = compute_psnr(simulated_video, gt_frame).mean()
                log_loss_dict["loss"].append(loss.item())
                log_loss_dict["l2_loss"].append(l2_loss.item())
                log_loss_dict["psnr"].append(psnr.item())
                log_loss_dict["ssim"].append(ssim_loss.item())

            # subtep-4: pass gradients to mpm solver

        nu_grad_norm = self.E_nu_list[1].grad.norm(2).item()
        spatial_grad_norm = 0
        for p in self.sim_fields.parameters():
            if p.grad is not None:
                spatial_grad_norm += p.grad.norm(2).item()
        velo_grad_norm = 0
        for p in self.velo_fields.parameters():
            if p.grad is not None:
                velo_grad_norm += p.grad.norm(2).item()

        renderd_video = torch.cat(rendered_video_list, dim=0)
        renderd_video = torch.clamp(renderd_video, 0.0, 1.0)
        visual_video = (renderd_video.detach().cpu().numpy() * 255.0).astype(np.uint8)
        gt_video = (gt_videos.detach().cpu().numpy() * 255.0).astype(np.uint8)

        if (
            self.step % self.gradient_accumulation_steps == 0
            or self.step == (self.train_iters - 1)
            or (self.step % self.log_iters == self.log_iters - 1)
        ):

            torch.nn.utils.clip_grad_norm_(
                self.trainable_params,
                self.max_grad_norm,
                error_if_nonfinite=False,
            )  # error if nonfinite is false

            self.optimizer.step()
            self.optimizer.zero_grad()
            if do_velo_opt:
                assert self.velo_optimizer is not None
                torch.nn.utils.clip_grad_norm_(
                    self.velo_fields.parameters(),
                    self.max_grad_norm * 10,
                    error_if_nonfinite=False,
                )  # error if nonfinite is false
                self.velo_optimizer.step()
                self.velo_optimizer.zero_grad()
                self.velo_scheduler.step()
            with torch.no_grad():
                self.E_nu_list[0].data.clamp_(1e-3, 2000)
                self.E_nu_list[1].data.clamp_(1e-2, 0.449)
        self.scheduler.step()

        for k, v in log_loss_dict.items():
            log_loss_dict[k] = np.mean(v)

        print(log_loss_dict)
        print(
            "nu: ",
            self.E_nu_list[1].item(),
            nu_grad_norm,
            spatial_grad_norm,
            velo_grad_norm,
            "young_mean, max:",
            youngs_padded.mean().item(),
            youngs_padded.max().item(),
            do_velo_opt,
        )

        if accelerator.is_main_process and (self.step % self.wandb_iters == 0):
            with torch.no_grad():
                wandb_dict = {
                    "nu_grad_norm": nu_grad_norm,
                    "spatial_grad_norm": spatial_grad_norm,
                    "velo_grad_norm": velo_grad_norm,
                    "nu": self.E_nu_list[1].item(),
                    # "mean_density": density.mean().item(),
                    "mean_E": youngs_padded.mean().item(),
                    "max_E": youngs_padded.max().item(),
                    "min_E": youngs_padded.min().item(),
                    "smoothness_loss": sm_loss.item(),
                    "window_size": window_size,
                    "velo_mean": init_velocity.mean().item(),
                    "velo_max": init_velocity.max().item(),
                }

                simulated_video = self.inference(cam)
                sim_video_torch = (
                    torch.from_numpy(simulated_video).float().to(device) / 255.0
                )
                gt_video_torch = torch.from_numpy(gt_video).float().to(device) / 255.0

                full_psnr = compute_psnr(sim_video_torch[1:], gt_video_torch)

                first_psnr = full_psnr[:6].mean().item()
                last_psnr = full_psnr[-6:].mean().item()
                full_psnr = full_psnr.mean().item()
                wandb_dict["full_psnr"] = full_psnr
                wandb_dict["first_psnr"] = first_psnr
                wandb_dict["last_psnr"] = last_psnr
                wandb_dict.update(log_loss_dict)

                if self.step % int(5 * self.wandb_iters) == 0:

                    wandb_dict["rendered_video"] = wandb.Video(
                        visual_video, fps=visual_video.shape[0]
                    )

                    wandb_dict["gt_video"] = wandb.Video(
                        gt_video,
                        fps=gt_video.shape[0],
                    )

                    wandb_dict["inference_video"] = wandb.Video(
                        simulated_video,
                        fps=simulated_video.shape[0],
                    )

                    simulated_video = self.inference(
                        cam, num_sec=3, substep=self.args.substep
                    )
                    wandb_dict["inference_video_t3"] = wandb.Video(
                        simulated_video,
                        fps=simulated_video.shape[0] // 3,
                    )

                    simulated_video = self.inference(
                        cam, velo_scaling=5.0, num_sec=3, substep=self.args.substep
                    )
                    wandb_dict["inference_video_v5_t3"] = wandb.Video(
                        simulated_video,
                        fps=simulated_video.shape[0] // 3,
                    )

                if self.use_wandb:
                    wandb.log(wandb_dict, step=self.step)

        self.accelerator.wait_for_everyone()

    def train(self):
        # might remove tqdm when multiple node
        for index in tqdm(range(self.step, self.train_iters), desc="Training progress"):
            self.train_one_step()
            if self.step % self.log_iters == self.log_iters - 1:
                if self.accelerator.is_main_process:
                    self.save()
                    # self.test()
            # self.accelerator.wait_for_everyone()
            self.step += 1
        if self.accelerator.is_main_process:
            self.save()

    @torch.no_grad()
    def inference(
        self, cam, velo_scaling=1.0, num_sec=1, nu=None, young_scaling=1.0, substep=20
    ):

        self.sim_fields.eval()
        self.velo_fields.eval()

        device = "cuda:{}".format(self.accelerator.process_index)

        time_stamps = np.linspace(0, 1, self.num_frames).astype(np.float32)[1:]
        time_idx = 0
        time_stamp = time_stamps[time_idx]

        density, youngs_padded, init_velocity, query_mask = self.get_density_velocity(
            time_stamp, device
        )
        youngs_padded = youngs_padded * young_scaling
        init_xyzs = self.particle_init_position

        padded_velocity = torch.zeros_like(init_xyzs)
        padded_velocity[query_mask, :] = init_velocity * velo_scaling

        num_particles = init_xyzs.shape[0]

        delta_time = 1.0 / (self.num_frames - 1)
        delta_time = 1.0 / 30
        substep_size = delta_time / substep
        num_substeps = int(delta_time / substep_size)
        # reset state
        self.set_simulation_state(
            init_xyzs,
            padded_velocity,
            device,
            requires_grad=True,
            use_precompute_F=False,
            use_density=False,
        )

        if nu is None:
            E, nu = self.E_nu_list[0].item(), self.E_nu_list[1].item()
        E_wp = from_torch_safe(youngs_padded, dtype=wp.float32, requires_grad=False)
        self.mpm_solver.set_E_nu(self.mpm_model, E_wp, nu, device=device)
        self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device=device)

        wp.launch(
            kernel=get_float_array_product,
            dim=num_particles,
            inputs=[
                self.mpm_state.particle_density,
                self.mpm_state.particle_vol,
                self.mpm_state.particle_mass,
            ],
            device=device,
        )

        pos_list = [self.particle_init_position.clone() * self.scale - self.shift]

        for i in tqdm(range((self.num_frames - 1) * num_sec)):
            for substep in range(num_substeps):
                self.mpm_solver.p2g2p(
                    self.mpm_model,
                    self.mpm_state,
                    substep,
                    substep_size,
                    device="cuda:0",
                )

            pos = wp.to_torch(self.mpm_state.particle_x).clone()
            pos = (pos * self.scale) - self.shift
            pos_list.append(pos)

        init_pos = pos_list[0].clone()
        pos_diff_list = [_ - init_pos for _ in pos_list]

        video_array = render_gaussian_seq_w_mask_with_disp(
            cam,
            self.render_params,
            init_pos,
            self.top_k_index,
            pos_diff_list,
            self.sim_mask_in_raw_gaussian,
        )

        video_numpy = video_array.detach().cpu().numpy() * 255
        video_numpy = np.clip(video_numpy, 0, 255).astype(np.uint8)

        return video_numpy

    def save(
        self,
    ):
        # training states
        output_path = os.path.join(
            self.output_path, f"checkpoint_model_{self.step:06d}"
        )
        os.makedirs(output_path, exist_ok=True)

        name_list = [
            "velo_fields",
            "sim_fields",
        ]
        for i, model in enumerate(
            [
                self.accelerator.unwrap_model(self.velo_fields, keep_fp32_wrapper=True),
                self.accelerator.unwrap_model(self.sim_fields, keep_fp32_wrapper=True),
            ]
        ):
            model_name = name_list[i]
            model_path = os.path.join(output_path, model_name + ".pt")
            torch.save(model.state_dict(), model_path)

    def load(self, checkpoint_dir):
        name_list = [
            "velo_fields",
            "sim_fields",
        ]
        for i, model in enumerate([self.velo_fields, self.sim_fields]):
            model_name = name_list[i]
            if model_name == "sim_fields" and (not self.args.run_eval):
                continue
            model_path = os.path.join(checkpoint_dir, model_name + ".pt")
            model.load_state_dict(torch.load(model_path))
            print("=> loaded: ", model_path)

    def setup_eval(self, args, gaussian_path, white_background=True):
        # setup gaussians
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
        camera_list = self.dataset.test_camera_list

        gaussians.load_ply(gaussian_path)
        gaussians.detach_grad()
        print(
            "load gaussians from: {}".format(gaussian_path),
            "... num gaussians: ",
            gaussians._xyz.shape[0],
        )
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_pipe = RenderPipe()

        render_params = RenderParams(
            render_pipe=render_pipe,
            bg_color=background,
            gaussians=gaussians,
            camera_list=camera_list,
        )
        self.render_params = render_params

        # get_gaussian scene box
        scaler = 1.1
        points = gaussians._xyz

        min_xyz = torch.min(points, dim=0)[0]
        max_xyz = torch.max(points, dim=0)[0]

        center = (min_xyz + max_xyz) / 2

        scaled_min_xyz = (min_xyz - center) * scaler + center
        scaled_max_xyz = (max_xyz - center) * scaler + center

        aabb = torch.stack([scaled_min_xyz, scaled_max_xyz], dim=0)

        # add filled in points
        gaussian_dir = os.path.dirname(gaussian_path)

        clean_points_path = os.path.join(gaussian_dir, "clean_object_points.ply")
        if os.path.exists(clean_points_path):
            clean_xyzs = pcu.load_mesh_v(clean_points_path)
            clean_xyzs = torch.from_numpy(clean_xyzs).float().to("cuda")
            self.clean_xyzs = clean_xyzs
            print(
                "loaded {} clean points from: ".format(clean_xyzs.shape[0]),
                clean_points_path,
            )
            # we can use tight threshold here
            not_sim_maks = find_far_points(
                gaussians._xyz, clean_xyzs, thres=0.01
            ).bool()
            sim_mask_in_raw_gaussian = torch.logical_not(not_sim_maks)
            # [N]
            self.sim_mask_in_raw_gaussian = sim_mask_in_raw_gaussian
        else:
            self.clean_xyzs = None
            self.sim_mask_in_raw_gaussian = torch.ones_like(gaussians._xyz[:, 0]).bool()

        return aabb

    def eval(
        self,
    ):

        accelerator = self.accelerator
        device = "cuda:{}".format(accelerator.process_index)
        data = next(self.dataloader)
        cam = data["cam"][0]

        nu = 0.1
        young_scaling = 5000.0
        substep = 800  # 1e-4
        video_numpy = self.inference(
            cam,
            velo_scaling=5.0,
            num_sec=3,
            nu=nu,
            young_scaling=young_scaling,
            substep=substep,
        )

        video_numpy = np.transpose(video_numpy, [0, 2, 3, 1])
        from motionrep.utils.io_utils import save_video_imageio, save_gif_imageio

        # output_dir = os.path.join(self.output_path, "simulation")
        output_dir = "./"

        save_path = os.path.join(
            output_dir,
            "eval_fill2k_video_nu_{}_ys_{}_substep_{}_grid_{}".format(
                nu, young_scaling, substep, self.args.grid_size
            )
            + ".gif",
        )
        print("save video to ", save_path)
        # save_video_imageio(save_path, video_numpy, fps=12)
        save_gif_imageio(save_path, video_numpy, fps=12)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")

    # dataset params
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../../data/physics_dreamer/alocasia_nerfstudio",
    )
    parser.add_argument(
        "--dataset_res",
        type=str,
        default="large",  # ["middle", "small", "large"]
    )

    parser.add_argument("--model", type=str, default="se3_field")
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_hidden_size", type=int, default=64)
    parser.add_argument("--spatial_res", type=int, default=32)
    parser.add_argument("--zero_init", type=bool, default=True)
    parser.add_argument("--entropy_cls", type=int, default=0)

    parser.add_argument("--num_frames", type=str, default=14)

    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--sim_res", type=int, default=24)
    parser.add_argument("--sim_output_dim", type=int, default=1)
    parser.add_argument("--substep", type=int, default=96)
    parser.add_argument("--loss_decay", type=float, default=1.0)
    parser.add_argument("--start_window_size", type=int, default=2)
    parser.add_argument("--compute_window", type=int, default=2)
    parser.add_argument("--grad_window", type=int, default=14)

    parser.add_argument("--downsample_scale", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=8)

    # loss parameters
    parser.add_argument("--tv_loss_weight", type=float, default=1e-2)
    parser.add_argument("--ssim", type=float, default=0.5)

    # Logging and checkpointing
    parser.add_argument("--output_dir", type=str, default="../../output/inverse_sim")
    parser.add_argument("--log_iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="path to load checkpoint from"
    )
    # training parameters
    parser.add_argument("--train_iters", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )

    # wandb parameters
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_entity", type=str, default="mit-cv")
    parser.add_argument("--wandb_project", type=str, default="inverse_sim")
    parser.add_argument("--wandb_iters", type=int, default=20)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--run_eval", action="store_true", default=False)

    # distributed training args
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    args, extra_args = parser.parse_known_args()
    cfg = create_config(args.config, args, extra_args)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    print(args.local_rank, "local rank")

    return cfg


if __name__ == "__main__":
    args = parse_args()

    # torch.backends.cuda.matmul.allow_tf32 = True

    trainer = Trainer(args)

    if args.run_eval:
        trainer.eval()
    else:
        # trainer.debug()
        trainer.train()
