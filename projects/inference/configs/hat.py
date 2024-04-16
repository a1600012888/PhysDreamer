import numpy as np

dataset_dir = "../../data/physics_dreamer/hat_nerfstudio/"
result_dir = "output/hat/demo"
exp_name = "hat"

model_list = [
    "../../../model/physdreamer/hat/model/",
]

focus_point_list = [
    np.array([-0.467188, 0.067178, 0.044333]),
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
    "init_young": 1e5,
    "downsample_scale": 0.04,
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
