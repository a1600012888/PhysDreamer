import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import point_cloud_utils as pcu
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs
import numpy as np
import logging
import argparse
import torch
import os
from physdreamer.utils.config import create_config
import numpy as np

from physdreamer.gaussian_3d.scene import GaussianModel

from physdreamer.data.datasets.multiview_dataset import MultiviewImageDataset
from physdreamer.data.datasets.multiview_video_dataset import (
    MultiviewVideoDataset,
    camera_dataset_collate_fn,
)

from physdreamer.data.datasets.multiview_dataset import (
    camera_dataset_collate_fn as camera_dataset_collate_fn_img,
)

from typing import NamedTuple

from physdreamer.utils.img_utils import compute_psnr, compute_ssim
from physdreamer.warp_mpm.mpm_data_structure import (
    MPMStateStruct,
    MPMModelStruct,
)
from physdreamer.warp_mpm.mpm_solver_diff import MPMWARPDiff
from physdreamer.warp_mpm.gaussian_sim_utils import get_volume
import warp as wp

from local_utils import (
    cycle,
    create_spatial_fields,
    find_far_points,
    apply_grid_bc_w_freeze_pts,
    add_constant_force,
    downsample_with_kmeans_gpu,
    render_gaussian_seq_w_mask_with_disp,
    render_gaussian_seq_w_mask_cam_seq_with_force_with_disp,
    get_camera_trajectory,
    render_gaussian_seq_w_mask_with_disp_for_figure,
)
from config_demo import DemoParams
from physdreamer.utils.io_utils import save_video_mediapy


logger = get_logger(__name__, log_level="INFO")


def create_dataset(args):

    res = [576, 1024]
    video_dir_name = "videos"

    # dataset = MultiviewVideoDataset(
    #     args.dataset_dir,
    #     use_white_background=False,
    #     resolution=res,
    #     scale_x_angle=1.0,
    #     video_dir_name=video_dir_name,
    # )
    dataset = MultiviewImageDataset(
        args.dataset_dir,
        use_white_background=False,
        resolution=res,
        scale_x_angle=1.0,
        load_imgs=False,
    )

    test_dataset = MultiviewImageDataset(
        args.dataset_dir,
        use_white_background=False,
        resolution=res,
        # use_index=[0],
        scale_x_angle=1.0,
        fitler_with_renderd=False,
        load_imgs=False,
    )
    print("len of test dataset", len(test_dataset))
    return dataset, test_dataset


class Trainer:
    def __init__(self, args):
        self.args = args

        logging_dir = os.path.join(args.output_dir, "debug_demo")
        accelerator_project_config = ProjectConfiguration(logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            mixed_precision="no",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
        )
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        set_seed(args.seed + accelerator.process_index)

        demo_cfg = DemoParams(args.scene_name).get_cfg(
            args.demo_name,
            args.model_id,
            args.eval_ys,
            args.force_id,
            args.force_mag,
            args.velo_scaling,
            args.point_id,
            args.cam_id,
            args.apply_force,
        )
        self.args.dataset_dir = demo_cfg["dataset_dir"]
        self.demo_cfg = demo_cfg

        # setup the dataset
        dataset, test_dataset = create_dataset(args)
        # will be used when synthesize camera trajectory
        self.test_dataset = test_dataset
        self.dataset = dataset
        dataset_dir = test_dataset.data_dir

        gaussian_path = os.path.join(dataset_dir, "point_cloud.ply")
        self.setup_render(
            args,
            gaussian_path,
            white_background=True,
        )
        self.args.substep = demo_cfg["substep"]
        self.args.grid_size = demo_cfg["grid_size"]
        self.args.checkpoint_path = demo_cfg["model_path"]
        self.demo_cfg = demo_cfg

        self.num_frames = int(args.num_frames)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            # collate_fn=camera_dataset_collate_fn,
            collate_fn=camera_dataset_collate_fn_img,
        )
        dataloader = accelerator.prepare(dataloader)
        # why be used in self.compute_metric
        self.dataloader = cycle(dataloader)
        self.accelerator = accelerator

        # init traiable params
        E_nu_list = self.init_trainable_params()
        for p in E_nu_list:
            p.requires_grad = False
        self.E_nu_list = E_nu_list

        # init simulation enviroment
        self.setup_simulation(dataset_dir, grid_size=args.grid_size)

        if args.checkpoint_path == "None":
            args.checkpoint_path = None
        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)
        self.sim_fields, self.velo_fields = accelerator.prepare(
            self.sim_fields, self.velo_fields
        )

    def init_trainable_params(
        self,
    ):
        # init young modulus and poisson ratio
        # from pre-optimized;  gres32 step 128.  300 epoch. lr 10.0  psnr: 27.72028086735652.  Stop at 100epoch
        young_numpy = np.array([self.demo_cfg["init_youngs"]]).astype(np.float32)
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
        """
        1. load internal filled points.
        2. pointcloud downsample with KMeans
        3. Setup MPM simulation environment
        """

        device = "cuda:{}".format(self.accelerator.process_index)

        xyzs = self.render_params.gaussians.get_xyz.detach().clone()
        sim_xyzs = xyzs[self.sim_mask_in_raw_gaussian, :]

        # scale, and shift
        pos_max = sim_xyzs.max()
        pos_min = sim_xyzs.min()
        scale = (pos_max - pos_min) * 1.8
        shift = -pos_min + (pos_max - pos_min) * 0.25
        self.scale, self.shift = scale, shift
        print("scale, shift", scale, shift)

        # load internal filled points.
        #   if exists, we will use it to fill in the internal points, but not for rendering
        #   we keep track of render_mask_in_sim_pts, to distinguish the orignal points from the internal filled points
        filled_in_points_path = os.path.join(dataset_dir, "internal_filled_points.ply")
        if os.path.exists(filled_in_points_path):
            fill_xyzs = pcu.load_mesh_v(filled_in_points_path)  # [n, 3]
            fill_xyzs = fill_xyzs[
                np.random.choice(
                    fill_xyzs.shape[0], int(fill_xyzs.shape[0] * 1.0), replace=False
                )
            ]
            fill_xyzs = torch.from_numpy(fill_xyzs).float().to("cuda")
            self.fill_xyzs = fill_xyzs
            print(
                "loaded {} internal filled points from: ".format(fill_xyzs.shape[0]),
                filled_in_points_path,
            )
            render_mask_in_sim_pts = torch.cat(
                [
                    torch.ones_like(sim_xyzs[:, 0]).bool(),
                    torch.zeros_like(fill_xyzs[:, 0]).bool(),
                ],
                dim=0,
            ).to(device)
            sim_xyzs = torch.cat([sim_xyzs, fill_xyzs], dim=0)
            self.render_mask = render_mask_in_sim_pts
        else:
            self.fill_xyzs = None
            self.render_mask = torch.ones_like(sim_xyzs[:, 0]).bool().to(device)

        sim_xyzs = (sim_xyzs + shift) / scale
        sim_aabb = torch.stack(
            [torch.min(sim_xyzs, dim=0)[0], torch.max(sim_xyzs, dim=0)[0]], dim=0
        )
        # This AABB is used to constraint the material fields and velocity fields.
        sim_aabb = (
            sim_aabb - torch.mean(sim_aabb, dim=0, keepdim=True)
        ) * 1.2 + torch.mean(sim_aabb, dim=0, keepdim=True)

        print("simulation aabb: ", sim_aabb)

        # point cloud resample with kmeans
        if "downsample_scale" in self.demo_cfg:
            downsample_scale = self.demo_cfg["downsample_scale"]
        else:
            downsample_scale = args.downsample_scale
        if downsample_scale > 0 and downsample_scale < 1.0:
            print("Downsample with ratio: ", downsample_scale)
            num_cluster = int(sim_xyzs.shape[0] * downsample_scale)

            # WARNING: this is a GPU implementation, and will be OOM if the number of points is large
            # you might want to use a CPU implementation if the number of points is large
            # For CPU implementation: uncomment the following lines
            # from local_utils import downsample_with_kmeans
            # sim_xyzs = downsample_with_kmeans(sim_xyzs.detach().cpu().numpy(), num_cluster)
            # sim_xyzs = torch.from_numpy(sim_xyzs).float().to(device)

            sim_xyzs = downsample_with_kmeans_gpu(sim_xyzs, num_cluster)

            sim_gaussian_pos = self.render_params.gaussians.get_xyz.detach().clone()[
                self.sim_mask_in_raw_gaussian, :
            ]
            sim_gaussian_pos = (sim_gaussian_pos + shift) / scale

            # record top k index for each point, to interpolate positions and rotations later
            cdist = torch.cdist(sim_gaussian_pos, sim_xyzs) * -1.0
            _, top_k_index = torch.topk(cdist, self.args.top_k, dim=-1)
            self.top_k_index = top_k_index

            print("Downsampled to: ", sim_xyzs.shape[0], "by", downsample_scale)

        # Compute the volume of each particle
        points_volume = get_volume(sim_xyzs.detach().cpu().numpy())

        num_particles = sim_xyzs.shape[0]
        sim_aabb = torch.stack(
            [torch.min(sim_xyzs, dim=0)[0], torch.max(sim_xyzs, dim=0)[0]], dim=0
        )
        sim_aabb = (
            sim_aabb - torch.mean(sim_aabb, dim=0, keepdim=True)
        ) * 1.2 + torch.mean(sim_aabb, dim=0, keepdim=True)

        # Initialize MPM state and model
        wp.init()
        wp.config.mode = "debug"
        wp.config.verify_cuda = True

        mpm_state = MPMStateStruct()
        mpm_state.init(num_particles, device=device, requires_grad=False)

        self.particle_init_position = sim_xyzs.clone()

        mpm_state.from_torch(
            self.particle_init_position.clone(),
            torch.from_numpy(points_volume).float().to(device).clone(),
            None,
            device=device,
            requires_grad=False,
            n_grid=grid_size,
            grid_lim=1.0,
        )
        mpm_model = MPMModelStruct()
        mpm_model.init(num_particles, device=device, requires_grad=False)
        # grid from [0.0 - 1.0]
        mpm_model.init_other_params(n_grid=grid_size, grid_lim=1.0, device=device)

        material_params = {
            # select from jel
            "material": "jelly",  # "jelly", "metal", "sand", "foam", "snow", "plasticine", "neo-hookean"
            "g": [0.0, 0.0, 0.0],
            "density": 2000,  # kg / m^3
            "grid_v_damping_scale": 1.1,  # no damping if > 1.0
        }
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
        assert os.path.exists(
            moving_pts_path
        ), "We need to segment out the moving part to initialize the boundary condition"

        moving_pts = pcu.load_mesh_v(moving_pts_path)
        moving_pts = torch.from_numpy(moving_pts).float().to(device)
        moving_pts = (moving_pts + shift) / scale
        freeze_mask = find_far_points(
            sim_xyzs, moving_pts, thres=0.5 / grid_size
        ).bool()
        freeze_pts = sim_xyzs[freeze_mask, :]

        grid_freeze_mask = apply_grid_bc_w_freeze_pts(
            grid_size, 1.0, freeze_pts, mpm_solver
        )
        self.freeze_mask = freeze_mask

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

    def add_constant_force(self, center_point, radius, force, dt, start_time, end_time):
        xyzs = self.particle_init_position.clone() * self.scale - self.shift

        device = "cuda:{}".format(self.accelerator.process_index)
        add_constant_force(
            self.mpm_solver,
            self.mpm_state,
            xyzs,
            center_point,
            radius,
            force,
            dt,
            start_time,
            end_time,
            device=device,
        )

    def get_simulation_input(self, device):
        """
        Outs: All padded
            density: [N]
            young_modulus: [N]
            poisson_ratio: [N]
            velocity: [N, 3]
            query_mask: [N]
            particle_F: [N, 3, 3]
            particle_C: [N, 3, 3]
        """

        density, youngs_modulus, ret_poisson = self.get_material_params(device)
        initial_position_time0 = self.particle_init_position.clone()

        query_mask = torch.logical_not(self.freeze_mask)
        query_pts = initial_position_time0[query_mask, :]

        velocity = self.velo_fields(query_pts)[..., :3]

        # scaling lr is similar to scaling the learning rate of velocity fields.
        velocity = velocity * 0.1  # not padded yet
        ret_velocity = torch.zeros_like(initial_position_time0)
        ret_velocity[query_mask, :] = velocity

        # init F as Idensity Matrix, and C and Zero Matrix
        I_mat = torch.eye(3, dtype=torch.float32).to(device)
        particle_F = torch.repeat_interleave(
            I_mat[None, ...], initial_position_time0.shape[0], dim=0
        )
        particle_C = torch.zeros_like(particle_F)

        return (
            density,
            youngs_modulus,
            ret_poisson,
            ret_velocity,
            query_mask,
            particle_F,
            particle_C,
        )

    def get_material_params(self, device):
        """
        Outs:
            density: [N]
            young_modulus: [N]
            poisson_ratio: [N]
        """

        initial_position_time0 = self.particle_init_position.detach()

        # query the materials params of all particles
        query_pts = initial_position_time0

        sim_params = self.sim_fields(query_pts)

        # scale the output of the network, similar to scale the learning rate
        sim_params = sim_params * 1000
        youngs_modulus = self.young_modulus.detach().clone()
        youngs_modulus += sim_params[..., 0]

        # clamp youngs modulus
        youngs_modulus = torch.clamp(youngs_modulus, 1.0, 5e8)

        density = self.density.detach().clone()
        ret_poisson = self.poisson_ratio.detach().clone()

        return density, youngs_modulus, ret_poisson

    def load(self, checkpoint_dir):
        name_list = [
            "velo_fields",
            "sim_fields",
        ]
        for i, model in enumerate([self.velo_fields, self.sim_fields]):
            model_name = name_list[i]
            model_path = os.path.join(checkpoint_dir, model_name + ".pt")
            if os.path.exists(model_path):
                print("=> loading: ", model_path)
                model.load_state_dict(torch.load(model_path))
            else:
                print("=> not found: ", model_path)

    def setup_render(self, args, gaussian_path, white_background=True):
        """
        1. Load 3D Gaussians in gaussian_path
        2. Prepare rendering params in self.render_params
        3. Load foreground points stored in the same directory as gaussian_path, with name "clean_object_points.ply"
               Only foreground points is used for simulation.
               We will track foreground points with mask: self.sim_mask_in_raw_gaussian
        """

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

        # segment foreground objects. Foreground points is stored in "clean_object_points.ply",
        #    only foreground points is used for simulation
        #    we will track foreground points with mask: self.sim_mask_in_raw_gaussian
        gaussian_dir = os.path.dirname(gaussian_path)

        clean_points_path = os.path.join(gaussian_dir, "clean_object_points.ply")

        assert os.path.exists(
            clean_points_path
        ), "We need to segment out the forground points to initialize the simulation"

        clean_xyzs = pcu.load_mesh_v(clean_points_path)
        clean_xyzs = torch.from_numpy(clean_xyzs).float().to("cuda")
        self.clean_xyzs = clean_xyzs
        print(
            "loaded {} clean points from: ".format(clean_xyzs.shape[0]),
            clean_points_path,
        )
        not_sim_maks = find_far_points(gaussians._xyz, clean_xyzs, thres=0.01).bool()
        sim_mask_in_raw_gaussian = torch.logical_not(not_sim_maks)
        # [N]
        self.sim_mask_in_raw_gaussian = sim_mask_in_raw_gaussian

    @torch.no_grad()
    def demo(
        self,
        velo_scaling=5.0,
        num_sec=3.0,
        eval_ys=1.0,
        static_camera=False,
        apply_force=False,
        save_name="demo",
    ):

        result_dir = self.demo_cfg["result_dir"]
        if "eval_ys" in self.demo_cfg:
            eval_ys = self.demo_cfg["eval_ys"]
        if "velo_scaling" in self.demo_cfg:
            velo_scaling = self.demo_cfg["velo_scaling"]

        save_name = self.demo_cfg["name"]

        if save_name.startswith("baseline"):
            self.compute_metric(save_name, result_dir)
            return

        # avoid re-run for experiment with the same name
        os.makedirs(result_dir, exist_ok=True)
        pos_path = os.path.join(result_dir, save_name + "_pos.npy")
        if os.path.exists(pos_path):
            pos_array = np.load(pos_path)
        else:
            pos_array = None

        device = "cuda:0"
        data = next(self.dataloader)
        cam = data["cam"][0]

        substep = self.args.substep  # 1e-4

        youngs_modulus = None

        self.sim_fields.eval()
        self.velo_fields.eval()

        (
            density,
            youngs_modulus_,
            poisson,
            init_velocity,
            query_mask,
            particle_F,
            particle_C,
        ) = self.get_simulation_input(device)

        poisson = self.E_nu_list[1].detach().clone()  # override poisson

        if eval_ys < 10:
            youngs_modulus = youngs_modulus_
        else:
            # assign eval_ys to all particles
            youngs_modulus = torch.ones_like(youngs_modulus_) * eval_ys

        # step-1 Setup simulation parameters. External force, or initial velocity.
        #   if --apply_force, we will apply a constant force to points close to the force center
        #   otherwise, we will load the initial velocity from pretrained models, and scale it by velo_scaling.

        delta_time = 1.0 / 30  # 30 fps
        substep_size = delta_time / substep
        num_substeps = int(substep)

        init_xyzs = self.particle_init_position.clone()

        init_velocity[query_mask, :] = init_velocity[query_mask, :] * velo_scaling
        if apply_force:
            init_velocity = torch.zeros_like(init_velocity)

            center_point = (
                torch.from_numpy(self.demo_cfg["center_point"]).to(device).float()
            )
            force = torch.from_numpy(self.demo_cfg["force"]).to(device).float()

            force_duration = self.demo_cfg["force_duration"]  # sec
            force_duration_steps = int(force_duration / delta_time)

            # apply force to points within the radius of the center point
            force_radius = self.demo_cfg["force_radius"]

            self.add_constant_force(
                center_point, force_radius, force, delta_time, 0.0, force_duration
            )

            # prepare to render force in simulated videos:
            #   find the closest point to the force center, and will use it to render the force
            xyzs = self.render_params.gaussians.get_xyz.detach().clone()
            dist = torch.norm(xyzs - center_point.unsqueeze(dim=0), dim=-1)
            closest_idx = torch.argmin(dist)
            closest_xyz = xyzs[closest_idx, :]
            render_force = force / force.norm() * 0.1
            do_render_force = True
        else:
            do_render_force = False

        # step-3: simulation or load the simulated sequence computed before
        #   with the same scene_name and demo_name
        if pos_array is None or save_name == "debug":
            self.mpm_state.reset_density(
                density.clone(), query_mask, device, update_mass=True
            )
            self.mpm_solver.set_E_nu_from_torch(
                self.mpm_model, youngs_modulus.clone(), poisson.clone(), device
            )
            self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device)

            self.mpm_state.continue_from_torch(
                init_xyzs,
                init_velocity,
                particle_F,
                particle_C,
                device=device,
                requires_grad=False,
            )

            # record drive points sequence
            render_pos_list = [(init_xyzs.clone() * self.scale) - self.shift]
            prev_state = self.mpm_state
            for i in tqdm(range(int((30) * num_sec))):
                # iterate over substeps for each frame
                for substep_local in range(num_substeps):
                    next_state = prev_state.partial_clone(requires_grad=False)
                    self.mpm_solver.p2g2p_differentiable(
                        self.mpm_model,
                        prev_state,
                        next_state,
                        substep_size,
                        device=device,
                    )
                    prev_state = next_state

                pos = wp.to_torch(next_state.particle_x).clone()
                # undo scaling and shifting
                pos = (pos * self.scale) - self.shift
                render_pos_list.append(pos)

            # save the sequence of drive points
            numpy_pos = torch.stack(render_pos_list, dim=0).detach().cpu().numpy()
            np.save(pos_path, numpy_pos)
        else:
            render_pos_list = []
            for i in range(pos_array.shape[0]):
                pos = pos_array[i, ...]
                render_pos_list.append(torch.from_numpy(pos).to(device))

        num_pos = len(render_pos_list)
        init_pos = render_pos_list[0].clone()
        pos_diff_list = [_ - init_pos for _ in render_pos_list]

        if not static_camera:
            interpolated_cameras = get_camera_trajectory(
                cam, num_pos, self.demo_cfg["camera_cfg"], self.test_dataset
            )
        else:
            interpolated_cameras = [cam] * num_pos

        if not do_render_force:
            video_array, moving_part_video = (
                render_gaussian_seq_w_mask_with_disp_for_figure(
                    interpolated_cameras,
                    self.render_params,
                    init_pos,
                    self.top_k_index,
                    pos_diff_list,
                    self.sim_mask_in_raw_gaussian,
                )
            )
            video_numpy = video_array.detach().cpu().numpy() * 255
            video_numpy = np.clip(video_numpy, 0, 255).astype(np.uint8)
            video_numpy = np.transpose(video_numpy, [0, 2, 3, 1])

            moving_part_video = moving_part_video.detach().cpu().numpy() * 255
            moving_part_video = np.clip(moving_part_video, 0, 255).astype(np.uint8)
            moving_part_video = np.transpose(moving_part_video, [0, 2, 3, 1])
        else:
            video_numpy = render_gaussian_seq_w_mask_cam_seq_with_force_with_disp(
                interpolated_cameras,
                self.render_params,
                init_pos,
                self.top_k_index,
                pos_diff_list,
                self.sim_mask_in_raw_gaussian,
                closest_idx,
                render_force,
                force_duration_steps,
            )
            video_numpy = np.transpose(video_numpy, [0, 2, 3, 1])

        if not static_camera:
            save_name = (
                save_name
                + "_movingcamera"
                + "_camid_{}".format(self.demo_cfg["cam_id"])
            )

        save_name = save_name + "_" + self.demo_cfg["name"]
        save_path = os.path.join(result_dir, save_name + ".mp4")

        print("save video to ", save_path)
        save_video_mediapy(video_numpy, save_path, fps=30)

        # save_path = save_path.replace(".mp4", "_moving_part.mp4")
        # save_video_mediapy(moving_part_video, save_path, fps=30)

    def compute_metric(self, exp_name, result_dir):

        data = next(self.dataloader)
        cam = data["cam"][0]

        # step-2 simulation part
        substep = self.args.substep  # 1e-4
        self.sim_fields.eval()
        self.velo_fields.eval()
        device = "cuda:{}".format(self.accelerator.process_index)

        (
            density,
            youngs_modulus,
            poisson,
            init_velocity,
            query_mask,
            particle_F,
            particle_C,
        ) = self.get_simulation_input(device)

        poisson = self.E_nu_list[1].detach().clone()  # override poisson
        # delta_time = 1.0 / (self.num_frames - 1)
        delta_time = 1.0 / 30  # 30 fps
        substep_size = delta_time / substep
        num_substeps = int(delta_time / substep_size)

        init_xyzs = self.particle_init_position.clone()
        init_velocity[query_mask, :] = init_velocity[query_mask, :]

        self.mpm_state.reset_density(
            density.clone(), query_mask, device, update_mass=True
        )
        self.mpm_solver.set_E_nu_from_torch(
            self.mpm_model, youngs_modulus.clone(), poisson.clone(), device
        )
        self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device)

        self.mpm_state.continue_from_torch(
            init_xyzs,
            init_velocity,
            particle_F,
            particle_C,
            device=device,
            requires_grad=False,
        )

        pos_list = [(init_xyzs.clone() * self.scale) - self.shift]

        prev_state = self.mpm_state
        for i in tqdm(range(self.args.num_frames - 1)):
            for substep_local in range(num_substeps):
                next_state = prev_state.partial_clone(requires_grad=False)
                self.mpm_solver.p2g2p_differentiable(
                    self.mpm_model,
                    prev_state,
                    next_state,
                    substep_size,
                    device=device,
                )
                prev_state = next_state

            pos = wp.to_torch(next_state.particle_x).clone()

            # pos = self.mpm_solver.export_particle_x_to_torch().clone()
            pos = (pos * self.scale) - self.shift
            pos_list.append(pos)
        # setup the camera trajectories (copy the static camera for n frames)
        init_pos = pos_list[0].clone()
        pos_diff_list = [_ - init_pos for _ in pos_list]

        interpolated_cameras = [cam] * len(pos_list)

        video_array = render_gaussian_seq_w_mask_with_disp(
            interpolated_cameras,
            self.render_params,
            init_pos,
            self.top_k_index,
            pos_diff_list,
            self.sim_mask_in_raw_gaussian,
        )
        video_numpy = video_array.detach().cpu().numpy() * 255
        video_numpy = np.clip(video_numpy, 0, 255).astype(np.uint8)
        video_numpy = np.transpose(video_numpy, [0, 2, 3, 1])
        os.makedirs(result_dir, exist_ok=True)
        save_path = os.path.join(
            result_dir,
            exp_name
            + "_jelly_densi2k_video_substep_{}_grid_{}".format(
                substep, self.args.grid_size
            )
            + ".mp4",
        )
        save_path = save_path.replace(".gif", ".mp4")
        save_video_mediapy(video_numpy, save_path, fps=25)

        gt_videos = data["video_clip"][0, 0 : self.num_frames, ...]
        ssim = compute_ssim(video_array, gt_videos)
        psnr = compute_psnr(video_array, gt_videos)

        print("psnr for each frame: ", psnr)
        mean_psnr = psnr.mean().item()
        print("mean psnr: ", mean_psnr, "mean ssim: ", ssim.item())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="se3_field")
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_hidden_size", type=int, default=64)
    # resolution of velocity fields
    parser.add_argument("--spatial_res", type=int, default=32)
    parser.add_argument("--zero_init", type=bool, default=True)

    parser.add_argument("--num_frames", type=str, default=14)

    # resolution of material fields
    parser.add_argument("--sim_res", type=int, default=8)
    parser.add_argument("--sim_output_dim", type=int, default=1)

    parser.add_argument("--downsample_scale", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=8)

    # Logging and checkpointing
    parser.add_argument("--output_dir", type=str, default="../../output/inverse_sim")
    parser.add_argument("--seed", type=int, default=0)

    # demo parameters. related to parameters specified in configs/{scene_name}.py
    parser.add_argument("--scene_name", type=str, default="carnation")
    parser.add_argument("--demo_name", type=str, default="inference_demo")
    parser.add_argument("--model_id", type=int, default=0)

    # if eval_ys > 10. Then all the youngs modulus is set to eval_ys homogeneously
    parser.add_argument("--eval_ys", type=float, default=1.0)
    parser.add_argument("--force_id", type=int, default=1)
    parser.add_argument("--force_mag", type=float, default=1.0)
    parser.add_argument("--velo_scaling", type=float, default=5.0)
    parser.add_argument("--point_id", type=int, default=0)
    parser.add_argument("--apply_force", action="store_true", default=False)
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--static_camera", action="store_true", default=False)

    args, extra_args = parser.parse_known_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)

    trainer.demo(
        velo_scaling=args.velo_scaling,
        eval_ys=args.eval_ys,
        static_camera=args.static_camera,
        apply_force=args.apply_force,
        save_name=args.demo_name,
    )
