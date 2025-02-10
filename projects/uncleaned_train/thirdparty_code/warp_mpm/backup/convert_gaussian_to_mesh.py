import os
from random import gauss
from fire import Fire
from motionrep.gaussian_3d.scene import GaussianModel
import numpy as np
import torch


def convert_gaussian_to_mesh(gaussian_path, save_path=None):
    if save_path is None:
        dir_path = os.path.dirname(gaussian_path)
        save_path = os.path.join(dir_path, "gaussian_to_mesh.obj")

    gaussian_path = os.path.join(gaussian_path)

    gaussians = GaussianModel(3)

    gaussians.load_ply(gaussian_path)
    gaussians.detach_grad()
    print(
        "load gaussians from: {}".format(gaussian_path),
        "... num gaussians: ",
        gaussians._xyz.shape[0],
    )

    mesh = gaussians.extract_mesh(
        save_path, density_thresh=1, resolution=128, decimate_target=1e5
    )

    mesh.write(save_path)


def internal_filling(gaussian_path, save_path=None, resolution=64):
    if save_path is None:
        dir_path = os.path.dirname(gaussian_path)
        save_path = os.path.join(dir_path, "gaussian_internal_fill.ply")

    gaussians = GaussianModel(3)

    gaussians.load_ply(gaussian_path)
    gaussians.detach_grad()

    print(
        "load gaussians from: {}".format(gaussian_path),
        "... num gaussians: ",
        gaussians._xyz.shape[0],
    )

    # [res, res, res]
    occ = (
        gaussians.extract_fields(resolution=resolution, num_blocks=16, relax_ratio=1.5)
        .detach()
        .cpu()
        .numpy()
    )

    xyzs = gaussians._xyz.detach().cpu().numpy()

    center = gaussians.center.detach().cpu().numpy()
    scale = gaussians.scale # float
    xyzs = (xyzs - center) * scale # [-1.5, 1.5]?

    percentile = [82, 84, 86][1]

    # from IPython import embed
    # embed()

    thres = np.percentile(occ, percentile)
    print("density threshold: {:.2f} -- in percentile: {:.1f} ".format(thres, percentile))
    occ_large_thres = occ > thres
    # get the xyz of the occupied voxels
    # xyz = np.argwhere(occ)
    # normalize to [-1, 1]
    # xyz = xyz / (resolution - 1) * 2 - 1

    voxel_counts = np.zeros((resolution, resolution, resolution))

    points_xyzindex = ((xyzs + 1) / 2 * (resolution - 1)).astype(np.uint32)

    for x, y, z in points_xyzindex:
        voxel_counts[x, y, z] += 1
    
    add_points = np.logical_and(occ_large_thres, voxel_counts <= 1)

    add_xyz = np.argwhere(add_points).astype(np.float32)
    add_xyz = add_xyz / (resolution - 1) * 2 - 1

    all_xyz = np.concatenate([xyzs, add_xyz], axis=0)

    print("added points: ", add_xyz.shape[0])
    
    # save to ply
    import point_cloud_utils as pcu

    pcu.save_mesh_vf(save_path, all_xyz, np.zeros((0, 3), dtype=np.int32))

    add_path = os.path.join(os.path.dirname(save_path), "extra_filled_points.ply")
    pcu.save_mesh_vf(add_path, add_xyz, np.zeros((0, 3), dtype=np.int32))



if __name__ == "__main__":
    # Fire(convert_gaussian_to_mesh)

    Fire(internal_filling)
