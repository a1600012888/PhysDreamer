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

    return points_volume
