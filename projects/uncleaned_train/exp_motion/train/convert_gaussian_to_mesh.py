import os
from random import gauss
from fire import Fire
from motionrep.gaussian_3d.scene import GaussianModel
import numpy as np
import torch


def convert_gaussian_to_mesh(gaussian_path, thresh=0.1, save_path=None):
    if save_path is None:
        dir_path = os.path.dirname(gaussian_path)
        save_path = os.path.join(dir_path, "gaussian_to_mesh_thres_{}.obj".format(thresh))

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
        save_path, density_thresh=thresh, resolution=128, decimate_target=1e5
    )

    mesh.write(save_path)


def internal_filling(gaussian_path, thresh=2.0,  save_path=None, resolution=256, 
                     num_pts=4):
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
    # torch.linspace(-1, 1, resolution) for the coords
    # x[0] => -1,  x[resolution-1] = 1
    # x[i] = -1 + i * 2 / (resolution - 1)
    # index_x = (x[i] + 1) / 2 * (resolution - 1)
    occ = (
        gaussians.extract_fields(resolution=resolution, num_blocks=16, relax_ratio=1.5)
        .detach()
        .cpu()
        .numpy()
    )

    xyzs = gaussians._xyz.detach().cpu().numpy()

    center = gaussians.center.detach().cpu().numpy()
    scale = gaussians.scale # float
    xyzs = (xyzs - center) * scale # [-1, 1]?

    percentile = [95, 97, 99][1]

    # from IPython import embed
    # embed()

    thres_ = np.percentile(occ, percentile)
    print("density threshold: {:.5f} -- in percentile: {:.1f} ".format(thres_, percentile))
    occ_large_thres = occ > thresh
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
    add_xyz = add_xyz / (resolution - 1) * 2 - 1  # [x,y,z]_min of the unit cell.  randomly add points in the unit cell

    cell_width = 2.0 / (resolution - 1)

    # copy add_xyz "num_pts" times
    add_xyz = np.repeat(add_xyz, num_pts, axis=0)

    random_offset_within_cell = np.random.uniform(-cell_width / 2, cell_width / 2, size=add_xyz.shape)
    add_xyz += random_offset_within_cell

    all_xyz = np.concatenate([xyzs, add_xyz], axis=0)

    print("added points: ", add_xyz.shape[0])
    
    # save to ply
    import point_cloud_utils as pcu

    # pcu.save_mesh_vf(save_path, all_xyz, np.zeros((0, 3), dtype=np.int32))

    add_path = os.path.join(os.path.dirname(save_path), "extra_filled_points_thresh_{}.ply".format(thresh))
    pcu.save_mesh_v(add_path, add_xyz)

    

if __name__ == "__main__":
    Fire(convert_gaussian_to_mesh)
    # Fire(internal_filling)
