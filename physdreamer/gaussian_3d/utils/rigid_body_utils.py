import torch
import torch.nn.functional as F


def get_rigid_transform(A, B):
    """
    Estimate the rigid body transformation between two sets of 3D points.
    A and B are Nx3 matrices where each row is a 3D point.
    Returns a rotation matrix R and translation vector t.
    Args:
        A, B: [batch, N, 3] matrix of 3D points
    Outputs:
        R, t: [batch, 3, 3/1]
        target = R @ source (source shape [3, 1]) + t
    """
    assert A.shape == B.shape, "Input matrices must have the same shape"
    assert A.shape[-1] == 3, "Input matrices must have 3 columns (x, y, z coordinates)"

    # Compute centroids. [..., 1, 3]
    centroid_A = torch.mean(A, dim=-2, keepdim=True)
    centroid_B = torch.mean(B, dim=-2, keepdim=True)

    # Center the point sets
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Compute the cross-covariance matrix. [..., 3, 3]
    H = A_centered.transpose(-2, -1) @ B_centered

    # Compute the Singular Value Decomposition. Along last two dimensions
    U, S, Vt = torch.linalg.svd(H)

    # Compute the rotation matrix
    R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)

    # Ensure a right-handed coordinate system
    flip_mask = (torch.det(R) < 0) * -2.0 + 1.0
    # Vt[:, 2, :] *= flip_mask[..., None]

    # [N] => [N, 3]
    pad_flip_mask = torch.stack(
        [torch.ones_like(flip_mask), torch.ones_like(flip_mask), flip_mask], dim=-1
    )
    Vt = Vt * pad_flip_mask[..., None]

    # Compute the rotation matrix
    R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)

    # print(R.shape, centroid_A.shape, centroid_B.shape, flip_mask.shape)
    # Compute the translation
    t = centroid_B - (R @ centroid_A.transpose(-2, -1)).transpose(-2, -1)
    t = t.transpose(-2, -1)
    return R, t


def _test_rigid_transform():
    # Example usage:
    A = torch.tensor([[1, 2, 3], [4, 5, 6], [9, 8, 10], [10, -5, 1]]) * 1.0

    R_synthesized = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) * 1.0
    # init a random rotation matrix:

    B = (R_synthesized @ A.T).T + 2.0  # Just an example offset

    R, t = get_rigid_transform(A[None, ...], B[None, ...])
    print("Rotation matrix R:")
    print(R)
    print("\nTranslation vector t:")
    print(t)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    from pytorch3d. Based on trace_method like: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L205
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def quternion_to_matrix(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    from Pytorch3d
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    From pytorch3d
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    ret = torch.stack((ow, ox, oy, oz), -1)
    ret = standardize_quaternion(ret)
    return ret


def _test_matrix_to_quaternion():
    # init a random batch of quaternion
    r = torch.randn((10, 4)).cuda()

    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    q = standardize_quaternion(q)

    R = quternion_to_matrix(q)

    I_rec = R @ R.transpose(-2, -1)
    I_rec_error = torch.abs(I_rec - torch.eye(3, device="cuda")[None, ...]).max()

    q_recovered = matrix_to_quaternion(R)
    norm_ = torch.linalg.norm(q_recovered, dim=-1)
    q_recovered = q_recovered / norm_[..., None]
    q_recovered = standardize_quaternion(q_recovered)

    print(q_recovered.shape, q.shape, R.shape)

    rec = (q - q_recovered).abs().max()

    print("rotation to I error:", I_rec_error, "quant rec error: ", rec)


def _test_matrix_to_quaternion_2():
    R = (
        torch.tensor(
            [[[1, 0, 0], [0, -1, 0], [0, 0, -1]], [[1, 0, 0], [0, 0, 1], [0, -1, 0]]]
        )
        * 1.0
    )

    q_rec = matrix_to_quaternion(R.transpose(-2, -1))

    R_rec = quternion_to_matrix(q_rec)

    print(R_rec)


if __name__ == "__main__":
    # _test_rigid_transform()
    _test_matrix_to_quaternion()

    _test_matrix_to_quaternion_2()
