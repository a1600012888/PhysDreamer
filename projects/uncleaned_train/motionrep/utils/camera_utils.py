import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def generate_spiral_path(
    pose: np.ndarray,
    radius: float,
    lookat_pt: np.ndarray = np.array([0, 0, 0]),
    up: np.ndarray = np.array([0, 0, 1]),
    n_frames: int = 60,
    n_rots: int = 1,
    y_scale: float = 1.0,
) -> np.ndarray:
    """Calculates a forward facing spiral path for rendering."""
    x_axis = pose[:3, 0]
    y_axis = pose[:3, 1]
    campos = pose[:3, 3]

    render_poses = []
    for theta in np.linspace(0.0, 2 * np.pi * n_rots, n_frames, endpoint=False):
        t = (np.cos(theta) * x_axis + y_scale * np.sin(theta) * y_axis) * radius
        position = campos + t
        z_axis = position - lookat_pt
        new_pose = np.eye(4)
        new_pose[:3] = viewmatrix(z_axis, up, position)
        render_poses.append(new_pose)
    render_poses = np.stack(render_poses, axis=0)
    return render_poses
