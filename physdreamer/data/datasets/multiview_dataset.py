from copy import deepcopy
import os
import sys
from typing import Union, Tuple, Dict

from jaxtyping import Float

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from physdreamer.data.scene_box import SceneBox
from physdreamer.gaussian_3d.utils.rigid_body_utils import get_rigid_transform
from physdreamer.data.cameras import Camera, focal2fov, fov2focal
from physdreamer.utils.colmap_utils import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
import json
from PIL import Image
from tqdm import tqdm


def read_uint8_rgba(img_path, img_hw=None):
    image = Image.open(img_path)
    if img_hw is not None:
        image = image.resize((img_hw[1], img_hw[0]), Image.BILINEAR)
    im_data = np.array(image.convert("RGBA"))
    # [H, W, 4]
    return im_data


def readColmapCameras(
    cam_extrinsics, cam_intrinsics, images_folder, img_hw=None, scale_x_angle=1.0
):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        height = intr.height
        width = intr.width
        if img_hw is not None:
            # keep FovX not changed, change aspect ratio accrodingly
            height = int(img_hw[0] / img_hw[1] * width)
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            if scale_x_angle != 1.0:
                focal_length_x = focal_length_x * scale_x_angle
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            if scale_x_angle != 1.0:
                focal_length_x = focal_length_x * scale_x_angle
                focal_length_y = focal_length_y * scale_x_angle
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))

        if img_hw is not None:
            height, width = img_hw

        cam_info = Camera(
            R=R,
            T=T,
            FoVy=FovY,
            FoVx=FovX,
            img_path=image_path,
            img_hw=(height, width),
            timestamp=None,
            data_device="cuda",
        )

        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


class MultiviewImageDataset(Dataset):
    """
    Dataset for standard NeRF training data
        Static. Multiview dataset

        Output:
            camera, image, mask, and other metadata like flow, depth..
    """

    def __init__(
        self,
        data_dir,
        use_white_background=True,
        subsample_factor=1,
        resolution=[576, 1024],
        use_index=None,
        scale_x_angle=1.0,
        load_imgs=True,
        fitler_with_renderd=False,
        flip_xy=False,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.use_white_background = use_white_background
        self.subsample_factor = subsample_factor
        self.resolution = resolution
        self.scale_x_angle = scale_x_angle
        self.load_imgs = load_imgs
        self.fitler_with_renderd = fitler_with_renderd
        self.flip_xy = flip_xy
        self._parse_dataset(data_dir, load_imgs=load_imgs, flip_xy=flip_xy)

        if use_index is not None:
            use_index = [_ for _ in use_index if _ < len(self.camera_list)]
            self.camera_list = [self.camera_list[i] for i in use_index]
            if load_imgs:
                self.np_uint8_rgba_list = [
                    self.np_uint8_rgba_list[i] for i in use_index
                ]
            # self.test_camera_list = [self.test_camera_list[i] for i in use_index]
            # self.test_camera_list = self.test_camera_list
            self.dataset_len = len(self.camera_list)

    def _parse_dataset(self, data_dir, load_imgs=True, flip_xy=False):
        camera_transform_file = os.path.join(data_dir, "transforms_train.json")
        camera_transform_file_test = os.path.join(data_dir, "transforms_train.json")

        if os.path.exists(camera_transform_file):
            print("=> loading camera from blender format transforms_train.json")
            camera_list, meta_dict = self._read_camera_transforms(
                camera_transform_file, img_hw=self.resolution, flip_xy=flip_xy
            )
            # assume test_meta_dict is the same as meta_dict
            test_camera_list, _ = self._read_camera_transforms(
                camera_transform_file_test, img_hw=self.resolution, flip_xy=flip_xy
            )
        else:
            assert os.path.exists(
                os.path.join(data_dir, "sparse/0")
            ), "colmap sparse folder not found!"
            print("=> loading camera from colmap format")
            camera_list, meta_dict = self._read_camera_transforms_colmap(
                data_dir, image_dir="images", img_hw=self.resolution, eval=False
            )
            test_camera_list, _ = self._read_camera_transforms_colmap(
                data_dir, image_dir="images", img_hw=self.resolution, eval=True
            )

        # filter out cameras with invalid images
        if self.fitler_with_renderd:
            rendered_dir = os.path.join(data_dir, "rendered_images")
            camera_list = self.filter_camera_with_renderd_frames(
                camera_list, rendered_dir
            )

        if "num_frames" not in meta_dict:
            meta_dict["num_frames"] = len(camera_list)

        if self.subsample_factor > 1:
            num_frames = int(meta_dict["num_frames"] / self.subsample_factor)
            num_images = int(len(camera_list) / self.subsample_factor)
            camera_list = camera_list[:num_images]
            test_camera_list = test_camera_list[:num_images]
            meta_dict["num_frames"] = num_frames
            meta_dict["num_cameras"] = int(
                meta_dict["num_cameras"] / self.subsample_factor
            )

        self.meta_dict = meta_dict
        self.camera_list = camera_list
        self.test_camera_list = test_camera_list

        self._num_frames = meta_dict[
            "num_frames"
        ]  # number of timestamps in the dataset
        self.num_cameras = meta_dict["num_cameras"]
        self.dataset_len = len(self.camera_list)

        if self.load_imgs:
            self._preload_imgs(camera_list)

    def _preload_imgs(self, camera_list):
        np_uint8_rgba_list = []
        print("preloading images...")
        for cam in tqdm(camera_list):
            im_data = read_uint8_rgba(cam.img_path, self.resolution)

            np_uint8_rgba_list.append(im_data)

        if self.resolution is None:
            self.resolution = np_uint8_rgba_list[0].shape[:2]

        print(
            "img dtype: ",
            np_uint8_rgba_list[0].dtype,
            "num_frames: ",
            self.num_frames,
            "image resolution(h,w): ",
            self.resolution,
        )

        self.np_uint8_rgba_list = np_uint8_rgba_list

    def __len__(self):
        return self.dataset_len

    @property
    def num_frames(self):
        return self._num_frames

    def __getitem__(self, idx):
        cam = self.camera_list[idx]
        if self.load_imgs:
            rgba = self.np_uint8_rgba_list[idx]

            norm_data = rgba / 255.0

            img = norm_data[:, :, :3] * norm_data[:, :, 3:4]
            mask = norm_data[:, :, 3:4]

            mask = torch.from_numpy(mask.astype(np.float32)).permute(2, 0, 1)
            # shape convert from HWC to CHW
            img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
            # img = (img - 0.5) / 0.5

            ret_dict = {
                "img": img,  # [3, H, W] value in [0, 1]
                "mask": mask,  # [1, H, W], 1 for foreground, 0 for background
                "cam": cam,
            }
        else:
            ret_dict = {
                "cam": cam,
            }

        img_path = cam.img_path
        img_name = os.path.basename(img_path).split(".")[0]
        ret_dict["img_name"] = img_name

        return ret_dict

    def _read_camera_transforms(
        self,
        camera_transform_file,
        img_hw=None,
        flip_xy=False,
    ):
        with open(camera_transform_file, "r") as f:
            camera_transforms = json.load(f)

        camera_list = []

        frames = camera_transforms["frames"]
        fovx = camera_transforms["camera_angle_x"]

        if self.scale_x_angle != 1.0:
            fovx = fovx * self.scale_x_angle

        for frame in frames:
            # img_path = os.path.join(self.data_dir, frame["file_path"] + ".png")
            img_path = os.path.join(self.data_dir, frame["file_path"])
            if not (
                img_path.endswith(".png")
                or img_path.endswith(".jpg")
                or img_path.endswith(".JPG")
                or img_path.endswith(".jpeg")
            ):
                img_path += ".png"
            # normalized timestamp in [0, 1]
            if img_hw is None:
                image = Image.open(img_path)
                img_hw = image.size[::-1]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            if flip_xy:
                w2c = w2c[np.array([1, 0, 2, 3]), :]
                w2c[1, :] *= -1
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            height, width = img_hw
            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy
            FovX = fovx

            camera_list.append(
                Camera(
                    R=R,
                    T=T,
                    FoVy=FovY,
                    FoVx=FovX,
                    img_path=img_path,
                    img_hw=img_hw,
                    timestamp=None,
                    data_device="cuda",
                )
            )

        if "num_frames" not in camera_transforms:
            num_frames = len(camera_list)
        meta_dict = {
            "num_frames": num_frames,
            "num_cameras": len(camera_list),
            "img_hw": img_hw,
        }

        return camera_list, meta_dict

    def _read_camera_transforms_colmap(
        self,
        path,
        image_dir,
        img_hw=None,
        eval=False,
        llffhold=8,
        flip_xy=False,
    ):
        assert flip_xy == False, "flip_xy not supported for colmap dataset"
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        reading_dir = "images" if image_dir == None else image_dir
        cam_infos = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(path, reading_dir),
            img_hw=img_hw,
            scale_x_angle=self.scale_x_angle,
        )

        if eval:
            ret_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % llffhold == 0
            ]
        else:
            ret_cam_infos = cam_infos

        if img_hw is None:
            img_hw = ret_cam_infos[0].img_hw
        num_frames = len(ret_cam_infos)
        meta_dict = {
            "num_frames": num_frames,
            "num_cameras": len(ret_cam_infos),
            "img_hw": img_hw,
        }
        return ret_cam_infos, meta_dict

    def filter_camera_with_renderd_frames(self, cam_list, rendered_dir):
        rendered_img_names = [_ for _ in os.listdir(rendered_dir) if _.endswith(".png")]
        rendered_img_names = [_.split(".")[0] for _ in rendered_img_names]

        for cam in cam_list:
            img_name = os.path.basename(cam.img_path).split(".")[0]
            if img_name not in rendered_img_names:
                cam_list.remove(cam)
        return cam_list

    def save_camera_list(self, cam_list, save_path):
        # R=R,
        # T=T,
        # FoVy=FovY,
        # FoVx=FovX,
        # img_path=img_path,
        # image_height=image_height,
        # image_width=image_width,
        camera_list = []
        assert save_path.endswith(".json"), "save_path should be a json file"

        for cam in cam_list:
            cam_dict = {
                "R": cam.R.tolist(),
                "T": cam.T.tolist(),
                "FoVy": cam.FoVy,
                "FoVx": cam.FoVx,
                "img_path": cam.img_path,
                "image_height": cam.image_height,
                "image_width": cam.image_width,
            }
            camera_list.append(cam_dict)

        with open(save_path, "w") as f:
            json.dump(camera_list, f, indent=4)

    def interpolate_camera(self, filename1, filename2, num_frames):
        for cam in self.camera_list:
            img_name = os.path.basename(cam.img_path).split(".")[0]
            if filename1.startswith(img_name):
                cam1 = cam
            if filename2.startswith(img_name):
                cam2 = cam

        interpolated_cameras = cam1.interpolate(cam2, num_frames - 1)
        return interpolated_cameras


def camera_dataset_collate_fn(batch):
    ret_dict = {
        "cam": [],
        "img_name": [],
    }

    for key in batch[0].keys():
        if key == "cam":
            ret_dict[key].extend([item[key] for item in batch])
        elif key == "img_name":
            ret_dict[key].extend([item[key] for item in batch])
        elif key == "timestamp":
            ret_dict[key] = torch.tensor([item[key] for item in batch])
        else:
            ret_dict[key] = torch.stack([item[key] for item in batch], dim=0)

    return ret_dict


def create_camera(dataset_dir, save_path, *args):
    fname1, fname2, fname3, num_frames = args
    num_frames_each = int(num_frames / 3)

    dataset = MultiviewImageDataset(dataset_dir, load_imgs=False)

    cam_AB = dataset.interpolate_camera(fname1, fname2, num_frames_each)
    cam_BC = dataset.interpolate_camera(fname2, fname3, num_frames_each)
    cam_CA = dataset.interpolate_camera(fname3, fname1, num_frames_each)

    cam_list = cam_AB + cam_BC + cam_CA
    dataset.save_camera_list(cam_list, save_path)


def test_speed():
    dataset_dir = "../../../../../dataset/3D_capture/purple_branches_colmap"
    dataset_dir = "../../../../../dataset/physics_dreamer/llff_flower_undistorted"

    dataset = MultiviewImageDataset(dataset_dir)

    data = dataset[0]

    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            print(key, val.shape)
        else:
            print(key, type(val))


if __name__ == "__main__":
    from fire import Fire

    Fire(create_camera)
