import numpy as np

dataset_dir = "../../data/physics_dreamer/hat_nerfstudio/"
result_dir = "output/hat/results_force"
exp_name = "hat"

model_list = [
    # multiview 64 364
    "../../output/inverse_sim/fast_hat_videos2_sv64-384_init1e5decay_1.0_substep_384_se3_field_lr_0.03_tv_0.0001_iters_200_sw_6_cw_1/seed0/checkpoint_model_000019",
]

focus_point_list = [
    np.array([-0.467188, 0.067178, 0.044333]),  # botton of the background
]

camera_cfg_list = [
    {
        "type": "interpolation",
        "start_frame": "frame_00001.png",
        "end_frame": "frame_00187.png",  # or 91
    },
    # real captured viewpoint
    {
        "type": "interpolation",
        "start_frame": "frame_00217.png",
    },
    # other selected viewpoint
    {
        "type": "interpolation",
        "start_frame": "frame_00001.png",
    },
    # other selected viewpoint
    {
        "type": "interpolation",
        "start_frame": "frame_00001.png",
    },
    # other selected viewpoint
    {
        "type": "interpolation",
        "start_frame": "frame_00079.png",
    },
]

simulate_cfg = {
    "substep": 384,
    "grid_size": 64,
}


points_list = [
    np.array([-0.390069, 0.139051, -0.182607]),  # bottom of the hat
    np.array([-0.404391, 0.184975, -0.001585]),  # middle of the hat
    np.array([-0.289375, 0.034581, 0.062010]),  # left of the hat
    np.array([-0.352060, 0.105737, 0.009359]),  # center of the hat
]

force_directions = [
    np.array([1.0, 0.0, 0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([1.0, 0.0, 1.0]),
    np.array([1.0, 1.0, 0.0]),
    np.array([1.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 1.0]),
    np.array([1.0, 1.0, 1.0]),
]

force_directions = np.array(force_directions)
force_directions = force_directions / np.linalg.norm(force_directions, axis=1)[:, None]
