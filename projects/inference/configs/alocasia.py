import numpy as np

dataset_dir = "../../data/physics_dreamer/alocasia/"
result_dir = "output/alocasia/results"
exp_name = "alocasia"

model_list = ["../../models/physdreamer/alocasia/model"]

focus_point_list = [
    np.array([-1.242875, -0.468537, -0.251450]),  # botton of the background
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
    # real captured viewpoint
    {
        "type": "interpolation",
        "start_frame": "frame_00236.png",
    },
    # another viewpoint
    {
        "type": "interpolation",
        "start_frame": "frame_00006.png",
    },
    # another viewpoint
    {
        "type": "interpolation",
        "start_frame": "frame_00095.png",
    },
]

simulate_cfg = {
    "substep": 768,
    "grid_size": 64,
    "init_young": 1e6,
    "downsample_scale": 0.1,  # downsample the points to speed up the simulation
}


points_list = [
    np.array([-0.508607, -0.180955, -0.123896]),  # top of the big stem
    np.array([-0.462227, -0.259485, -0.112966]),  # top of the second stem
    np.array([-0.728061, -0.092306, -0.149104]),  # top of the third stem
    np.array([-0.603330 - 0.204207 - 0.127469]),  # top of the 4th stem
    np.array([-0.408097, -0.076293, -0.110391]),  # top of the big leaf
    np.array([-0.391575, -0.224018, -0.052054]),  # top of the second leaf
    np.array([-0.768167, -0.032502, -0.143995]),  # top of the third leaf
    np.array([-0.633866, -0.170207, -0.103671]),  # top of the 4th leaf
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
