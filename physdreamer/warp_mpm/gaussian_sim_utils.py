import numpy as np


def get_volume(xyzs: np.ndarray, resolution=128) -> np.ndarray:

    # set a grid in the range of [-1, 1], with resolution
    voxel_counts = np.zeros((resolution, resolution, resolution))

    points_xyzindex = ((xyzs + 1) / 2 * (resolution - 1)).astype(np.uint32)
    cell_volume = (2.0 / (resolution - 1)) ** 3

    for x, y, z in points_xyzindex:
        voxel_counts[x, y, z] += 1

    points_number_in_corresponding_voxel = voxel_counts[
        points_xyzindex[:, 0], points_xyzindex[:, 1], points_xyzindex[:, 2]
    ]

    points_volume = cell_volume / points_number_in_corresponding_voxel

    points_volume = points_volume.astype(np.float32)

    # some statistics
    num_non_empyt_voxels = np.sum(voxel_counts > 0)
    max_points_in_voxel = np.max(voxel_counts)
    min_points_in_voxel = np.min(voxel_counts)
    avg_points_in_voxel = np.sum(voxel_counts) / num_non_empyt_voxels
    print("Number of non-empty voxels: ", num_non_empyt_voxels)
    print("Max points in voxel: ", max_points_in_voxel)
    print("Min points in voxel: ", min_points_in_voxel)
    print("Avg points in voxel: ", avg_points_in_voxel)

    return points_volume
