import point_cloud_utils as pcu
import argparse
import os
import json
import numpy as np


def transform_vertex(vertex: np.ndarray, transform_dict):
    """
    Args:
        vertex: shape [n, 3]
    """
    if transform_dict is not None:
        center = np.array(transform_dict["center"])
        scale = transform_dict["scale"]

    else:
        center = np.mean(vertex, axis=0)
        scale = np.max(np.abs(vertex - center))

    new_vertex = (vertex - center) / scale

    return new_vertex, center, scale


def colmap_to_blender_transform(vertex: np.ndarray):
    R_mat = np.array(
        [[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]],
    )
    vertex = R_mat[np.newaxis, :, :] @ vertex[:, :, np.newaxis]

    return vertex.squeeze(axis=-1)


def copy_mtl_file(obj_path, transformed_obj_path):
    mtl_path = obj_path.replace(".obj", ".mtl")

    dummy_mtl_path = transformed_obj_path + ".mtl"
    if os.path.exists(dummy_mtl_path):
        os.remove(dummy_mtl_path)

    if os.path.exists(mtl_path):
        os.system("cp {} {}".format(mtl_path, dummy_mtl_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, required=True)

    parser.add_argument("--save_transform", action="store_true", default=False)

    args = parser.parse_args()

    dir_name = os.path.dirname(args.obj_path)
    _name = os.path.basename(dir_name)

    dir_name_father = os.path.dirname(dir_name)

    transformed_dir = os.path.join(dir_name_father, "transformed_{}".format(_name))
    if not os.path.exists(transformed_dir):
        os.makedirs(transformed_dir)

    transformed_obj_path = os.path.join(
        transformed_dir, os.path.basename(args.obj_path)
    )

    if os.path.exists(transformed_obj_path):
        print("Transformed object already exists.")
        # return

    meta_path = os.path.join(dir_name, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)
    else:
        print("transforming without meta.json")
        meta_dict = None

    mesh = pcu.load_triangle_mesh(args.obj_path)
    vertex = mesh.v
    vertex, center, scale = transform_vertex(vertex, meta_dict)
    vertex = colmap_to_blender_transform(vertex)

    mesh.vertex_data.positions = vertex

    mesh.save(transformed_obj_path)

    copy_mtl_file(args.obj_path, transformed_obj_path)

    if args.save_transform:
        transform_dict = {"center": center.tolist(), "scale": scale}
        with open(os.path.join(dir_name, "meta.json"), "w") as f:
            json.dump(transform_dict, f)

        print("Saved transform dict to {}".format(os.path.join(dir_name, "meta.json")))


if __name__ == "__main__":
    main()
