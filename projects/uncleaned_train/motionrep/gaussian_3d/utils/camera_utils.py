#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from motionrep.gaussian_3d.scene.cameras import Camera
import numpy as np
from motionrep.gaussian_3d.utils.general_utils import PILtoTorch
from motionrep.gaussian_3d.utils.graphics_utils import fov2focal
import torch

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry


def look_at(from_point, to_point, up_vector=(0, 1, 0)):
    """
    Compute the look-at matrix for a camera.

    :param from_point: The position of the camera.
    :param to_point: The point the camera is looking at.
    :param up_vector: The up direction of the camera.
    :return: The 4x4 look-at matrix.
    """

    # minus z for opengl. z for colmap
    forward = np.array(to_point) - np.array(from_point)
    forward = forward / (np.linalg.norm(forward) + 1e-5)

    # x-axis
    # Right direction is the cross product of the forward vector and the up vector
    right = np.cross(up_vector, forward)
    right = right / (np.linalg.norm(right) + 1e-5)

    # y axis
    # True up direction is the cross product of the right vector and the forward vector
    true_up = np.cross(forward, right)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-5)

    # camera to world
    rotation = np.array(
        [
            [right[0], true_up[0], forward[0]],
            [right[1], true_up[1], forward[1]],
            [right[2], true_up[2], forward[2]],
        ]
    )

    # Construct the translation matrix
    translation = np.array(
        [
            [-from_point[0]],
            [-from_point[1]],
            [-from_point[2]],
        ]
    )

    # Combine the rotation and translation to get the look-at matrix
    T = 1.0 * rotation.transpose() @ translation

    return rotation.transpose(), T


def create_cameras_around_sphere(
    radius=6,
    elevation=0,
    fovx=35,
    resolutions=(720, 1080),
    num_cams=60,
    center=(0, 0, 0),
):
    """
    Create cameras around a sphere.

    :param radius: The radius of the circle on which cameras are placed.
    :param elevation: The elevation angle in degrees.
    :param fovx: The horizontal field of view of the cameras.
    :param resolutions: The resolution of the cameras.
    :param num_cams: The number of cameras.
    :param center: The center of the sphere.
    :return: A list of camera extrinsics (world2camera transformations).
    """
    extrinsics = []

    # Convert elevation to radians
    elevation_rad = np.radians(elevation)

    # Compute the y-coordinate of the cameras based on the elevation
    z = radius * np.sin(elevation_rad)

    # Compute the radius of the circle at the given elevation
    circle_radius = radius * np.cos(elevation_rad)

    for i in range(num_cams):
        # Compute the angle for the current camera
        angle = 2 * np.pi * i / num_cams

        # Compute the x and z coordinates of the camera
        x = circle_radius * np.cos(angle) + center[0]
        y = circle_radius * np.sin(angle) + center[1]

        # Create the look-at matrix for the camera
        R, T = look_at((x, y, z + center[2]), center)
        extrinsics.append([R, T.squeeze(axis=-1)])

    cam_list = []
    dummy_image = torch.tensor(
        np.zeros((3, resolutions[0], resolutions[1]), dtype=np.uint8)
    )
    for i in range(num_cams):
        R, T = extrinsics[i]

        # R is stored transposed due to 'glm' in CUDA code
        R = R.transpose()
        cam = Camera(
            colmap_id=i,
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovx * resolutions[1] / resolutions[0],
            image_name="",
            uid=i,
            data_device="cuda",
            image=dummy_image,
            gt_alpha_mask=None,
        )

        cam_list.append(cam)

    return cam_list
