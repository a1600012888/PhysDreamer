import numpy as np

dataset_dir = "../../data/physics_dreamer/carnations/"
result_dir = "output/carnations/demos"
exp_name = "carnations"


model_list = [
    "../../models/physdreamer/carnations/model",
]

focus_point_list = [
    np.array([0.189558, 2.064228, -0.216089]),  # botton of the background
]

camera_cfg_list = [
    {
        "type": "spiral",
        "focus_point": focus_point_list[0],
        "radius": 0.05,
        "up": np.array([0, -0.5, 1]),
    },
    {
        "type": "interpolation",
        "start_frame": "frame_00001.png",
        "end_frame": "frame_00022.png",
    },
    # real capture viewpoint
    {
        "type": "interpolation",
        "start_frame": "frame_00219.png",
    },
    # another render viewpoint
    {
        "type": "interpolation",
        "start_frame": "frame_00106.png",
    },
    # another render viewpoint
    {
        "type": "interpolation",
        "start_frame": "frame_00011.png",
    },
]

simulate_cfg = {
    "substep": 768,
    "grid_size": 64,
    "init_young": 2140628.25,  # save the initialized young's modulus, since optimized
    "downsample_scale": 0.1,  # downsample the points to speed up the simulation
}


points_list = [
    np.array([0.076272, 0.848310, 0.074134]),  # top of the flower
    np.array([0.057208, 0.848147, -0.013685]),  # middle of the flower
    np.array([0.134908, 0.912759, -0.023763]),  # top of the stem
    np.array([0.169540, 0.968676, -0.095261]),  # middle of the stem
    np.array([0.186664, 1.028284, -0.187793]),  # bottom of the stem
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

force_directions_old_carnations = [
    np.array([2.0, 1.0, 0]),  # horizontal to left
    np.array([0.0, 1.0, 2.0]),  # vertical to top
    np.array([1.0, 1.0, 1.0]),  # top right to bottom left
    np.array([0.0, 1.0, 0.0]),  # orthgonal to the screen,
]
