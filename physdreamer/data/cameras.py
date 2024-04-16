import torch
import numpy as np
import torch.nn as nn
import math
from typing import Tuple
from scipy.spatial.transform import Rotation as R


class Camera(nn.Module):
    def __init__(
        self,
        R: np.ndarray,
        T: np.ndarray,
        FoVx,
        FoVy,
        img_path,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        img_hw: Tuple[int, int] = (800, 800),
        timestamp: float = 0.0,
    ):
        super(Camera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.img_path = img_path

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.image_width = img_hw[1]
        self.image_height = img_hw[0]
        self.time_stamp = timestamp

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale))
            .transpose(0, 1)
            .to(self.data_device)
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(self.data_device)
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # [2, 2].
        #  (w2c @ p) / depth => cam_plane
        #  (p_in_cam / depth)[:2] @  cam_plane_2_img => [pixel_x, pixel_y]    cam_plane => img_plane
        self.cam_plane_2_img = torch.tensor(
            [
                [0.5 * self.image_width / math.tan(self.FoVx / 2.0), 0.0],
                [0.0, 0.5 * self.image_height / math.tan(self.FoVy / 2.0)],
            ]
        ).to(self.data_device)

    def interpolate(self, next_camera, steps=24):
        interpolated_R_t = interpolate_cameras(
            self.R, self.T, next_camera.R, next_camera.T, steps
        )

        ret_camera_list = []

        if self.time_stamp is None:
            self.time_stamp = 0.0
        if next_camera.time_stamp is None:
            next_camera.time_stamp = 0.0

        for i, (R, t) in enumerate(interpolated_R_t):
            time_stamp = (
                self.time_stamp
                + (next_camera.time_stamp - self.time_stamp) * (i) / steps
            )
            cam = Camera(
                R,
                t,
                self.FoVx,
                self.FoVy,
                self.img_path,
                self.trans,
                self.scale,
                self.data_device,
                (self.image_height, self.image_width),
                time_stamp,
            )
            ret_camera_list.append(cam)

        return ret_camera_list


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def slerp_quaternions(q1, q2, t):
    """Perform spherical linear interpolation between two quaternions."""
    # Compute the cosine of the angle between the two vectors.
    dot = np.dot(q1, q2)

    # If the dot product is negative, the quaternions have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # Clamp dot product to be in the range of Acos()
    # This may be necessary when floating point errors occur
    dot = np.clip(dot, -1.0, 1.0)

    theta_0 = np.arccos(dot)  # theta_0 = angle between input vectors
    theta = theta_0 * t  # theta = angle between v0 and result

    q2_ = q2 - q1 * dot
    q2_ = q2_ / (np.linalg.norm(q2_) + 1e-6)

    return q1 * np.cos(theta) + q2_ * np.sin(theta)


def interpolate_cameras(R1, t1, R2, t2, steps=10):
    # 0 <= alpha <= 1; in total (steps + 1) cameras
    # Convert rotation matrices to quaternions

    q1 = R.from_matrix(R1).as_quat()
    q2 = R.from_matrix(R2).as_quat()

    interpolated_cameras = []
    for step in range(steps + 1):
        alpha = step / steps

        # Spherical linear interpolation of quaternions
        q_interp = slerp_quaternions(q1, q2, alpha)
        R_interp = R.from_quat(q_interp).as_matrix()

        # Linear interpolation of translation vectors
        t_interp = (1 - alpha) * t1 + alpha * t2

        interpolated_cameras.append((R_interp, t_interp))

    return interpolated_cameras
