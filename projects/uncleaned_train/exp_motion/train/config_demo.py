import numpy as np

from model_config import (
    model_list,
    camera_cfg_list,
    points_list,
    force_directions,
    simulate_cfg,
    dataset_dir,
    result_dir,
    exp_name,
)


class DemoParams(object):
    def __init__(self):

        self.demo_dict = {
            "baseline": {
                "model_path": model_list[0],
                "substep": 768,
                "grid_size": 64,
                "name": "baseline",
                "camera_cfg": camera_cfg_list[0],
                "cam_id": 0,
            },
            "demo_dummy": {
                "model_path": model_list[0],
                "center_point": points_list[0],
                "force": np.array([0.15, 0, 0]),
                "camera_cfg": camera_cfg_list[0],
                "force_duration": 0.75,
                "force_radius": 0.1,
                "substep": 256,
                "grid_size": 96,
                "total_time": 5,
                "name": "alocasia_sv_gres96_substep256_force_top_of_flower",
            },
        }

    def get_cfg(
        self,
        demo_name=None,
        model_id: int = 0,
        eval_ys: float = 1.0,
        force_id: int = 0,
        force_mag: float = 1.0,
        velo_scaling: float = 3.0,
        point_id: int = 0,
        cam_id: int = 0,
        apply_force: bool = False,
    ):
        if demo_name == "None":
            demo_name = None
        if (demo_name is not None) and (demo_name in self.demo_dict):
            cfg = self.demo_dict[demo_name]
        else:
            cfg = {}
            cfg["model_path"] = model_list[model_id]
            cfg["center_point"] = points_list[point_id]
            cfg["force"] = force_directions[force_id] * force_mag
            cfg["camera_cfg"] = camera_cfg_list[cam_id]
            cfg["cam_id"] = cam_id
            cfg["force_duration"] = 0.75
            cfg["force_radius"] = 0.1
            cfg["substep"] = simulate_cfg["substep"]
            cfg["grid_size"] = simulate_cfg["grid_size"]
            cfg["total_time"] = 5
            cfg["eval_ys"] = eval_ys
            cfg["velo_scaling"] = velo_scaling

            if demo_name is None:
                name = ""
            else:
                name = demo_name + "_"
            name = (
                name + f"{exp_name}_sv_gres{cfg['grid_size']}_substep{cfg['substep']}"
            )
            if eval_ys > 10:
                name += f"_eval_ys_{eval_ys}"
            else:
                name += f"_model_{model_id}"

            if apply_force:
                name += f"_force_{force_id}_mag_{force_mag}_point_{point_id}"
            else:
                name += f"_no_force_velo_{velo_scaling}"
            cfg["name"] = name

        cfg["dataset_dir"] = dataset_dir
        cfg["result_dir"] = result_dir

        return cfg
