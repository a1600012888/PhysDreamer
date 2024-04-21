import numpy as np

exp_name = "telephone"
dataset_dir = "../../data/physics_dreamer/telephone/"
result_dir = "output/telephone/results"

model_list = ["../../models/physdreamer/telephone/model"]

focus_point_list = [
    np.array([-0.401468, 0.889287, -0.116852]),  # botton of the background
]

camera_cfg_list = [
    {
        "type": "spiral",
        "focus_point": focus_point_list[0],
        "radius": 0.1,
        "up": np.array([0, 0, 1]),
    },
    {
        "type": "interpolation",
        "start_frame": "frame_00001.png",
        "end_frame": "frame_00019.png",
    },
    # real video viewpoint
    {
        "type": "interpolation",
        "start_frame": "frame_00190.png",
    },
    # other selected viewpoint
    {
        "type": "interpolation",
        "start_frame": "frame_00037.png",
    },
    # other selected viewpoint
    {
        "type": "interpolation",
        "start_frame": "frame_00090.png",
    },
]

simulate_cfg = {
    "substep": 256,
    "grid_size": 96,
    "init_young": 1e5,
    "downsample_scale": 0.1,  # downsample the points to speed up the simulation
}


points_list = [
    np.array([-0.417240, 0.907780, -0.379144]),  # bottom of the lines.
    np.array([-0.374907, 0.796209, -0.178907]),  # middle of the right lines
    np.array([-0.414156, 0.901207, -0.182275]),  # middle of the left lines
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
