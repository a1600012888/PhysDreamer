import numpy as np

# from model_config import (
#     model_list,
#     camera_cfg_list,
#     points_list,
#     force_directions,
#     simulate_cfg,
#     dataset_dir,
#     result_dir,
#     exp_name,
# )
import importlib.util
import os


class DemoParams(object):
    def __init__(self, scene_name):

        self.scene_name = scene_name
        base_dir = os.path.dirname(__file__)

        # import_file_path = ".configs." + scene_name
        import_file_path = os.path.join(base_dir, "configs", scene_name + ".py")
        print("loading scene params from: ", import_file_path)
        spec = importlib.util.spec_from_file_location(scene_name, import_file_path)
        if spec is None:
            print(f"Could not load the spec for: {import_file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.model_list = module.model_list
        self.camera_cfg_list = module.camera_cfg_list
        self.points_list = module.points_list
        self.force_directions = module.force_directions
        self.simulate_cfg = module.simulate_cfg
        self.dataset_dir = module.dataset_dir
        self.result_dir = module.result_dir
        self.exp_name = module.exp_name

        substep = self.simulate_cfg["substep"]
        grid_size = self.simulate_cfg["grid_size"]
        self.init_youngs = self.simulate_cfg["init_young"]
        self.downsample_scale = self.simulate_cfg["downsample_scale"]

        self.demo_dict = {
            "baseline": {
                "model_path": self.model_list[0],
                "substep": substep,
                "grid_size": grid_size,
                "name": "baseline",
                "camera_cfg": self.camera_cfg_list[0],
                "cam_id": 0,
                "init_youngs": self.init_youngs,
                "downsample_scale": self.downsample_scale,
            }
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
            cfg["model_path"] = self.model_list[model_id]
            cfg["center_point"] = self.points_list[point_id]
            cfg["force"] = self.force_directions[force_id] * force_mag
            cfg["camera_cfg"] = self.camera_cfg_list[cam_id]
            cfg["cam_id"] = cam_id
            cfg["force_duration"] = 0.75
            cfg["force_radius"] = 0.1
            cfg["substep"] = self.simulate_cfg["substep"]
            cfg["grid_size"] = self.simulate_cfg["grid_size"]
            cfg["total_time"] = 5
            cfg["eval_ys"] = eval_ys
            cfg["velo_scaling"] = velo_scaling

            if demo_name is None:
                name = ""
            else:
                name = demo_name + "_"
            name = (
                name
                + f"{self.scene_name}_sv_gres{cfg['grid_size']}_substep{cfg['substep']}"
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

        cfg["dataset_dir"] = self.dataset_dir
        cfg["result_dir"] = self.result_dir
        cfg["init_youngs"] = self.init_youngs
        cfg["downsample_scale"] = self.downsample_scale

        return cfg
