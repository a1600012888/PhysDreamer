from copy import deepcopy
import os
from typing import Union, Tuple, Dict

from jaxtyping import Float

import numpy as np
import random
import torch
from decord import VideoReader
import torchvision.transforms as transforms
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
import sys


def read_uint8_rgba(img_path, img_hw=None):
    if not (img_path.endswith(".png") or img_path.endswith(".JPG")):
        img_path = img_path + ".png"

    image = Image.open(img_path)
    if img_hw is not None:
        image = image.resize((img_hw[1], img_hw[0]), Image.BILINEAR)
    im_data = np.array(image.convert("RGBA"))
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


class MultiviewVideoDataset(Dataset):
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
        resolution=[576, 1024],
        scale_x_angle=1.0,
        use_index=None,
        video_dir_name="videos",
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.video_dir = os.path.join(data_dir, video_dir_name)
        assert os.path.exists(self.video_dir), "videos not exist!"
        self.use_white_background = use_white_background
        self.resolution = resolution
        self.scale_x_angle = scale_x_angle
        self._parse_dataset(data_dir)

        self.video_transforms = transforms.Compose([])
        if use_index is not None:
            use_index = [_ for _ in use_index if _ < len(self.camera_list)]
            self.camera_list = [self.camera_list[i] for i in use_index]
            self.np_uint8_rgba_list = [self.np_uint8_rgba_list[i] for i in use_index]
            # self.test_camera_list = [self.test_camera_list[i] for i in use_index]
            # self.test_camera_list = self.test_camera_list
            self.dataset_len = len(self.camera_list)

        mask_dir = os.path.join(data_dir, "masks")
        self.mask_float32_list = []
        # reed masks
        for cam in self.camera_list:
            img_name = os.path.basename(cam.img_path).split(".")[0]
            mask_path = os.path.join(mask_dir, img_name + "_mask.png")
            if os.path.exists(mask_path):
                mask = read_uint8_rgba(mask_path, self.resolution)[..., 0:1]
                mask = mask / 255.0
                mask = mask.astype(np.float32)
            else:
                mask = np.ones(
                    (self.resolution[0], self.resolution[1], 1), dtype=np.float32
                )
            self.mask_float32_list.append(mask)

    def _parse_dataset(self, data_dir):
        camera_transform_file = os.path.join(data_dir, "transforms_train.json")
        camera_transform_file_test = os.path.join(data_dir, "transforms_train.json")

        if os.path.exists(camera_transform_file):
            print("=> loading camera from blender format transforms_train.json")
            camera_list, meta_dict = self._read_camera_transforms(
                camera_transform_file, img_hw=self.resolution
            )
            # assume test_meta_dict is the same as meta_dict
            test_camera_list, _ = self._read_camera_transforms(
                camera_transform_file_test, img_hw=self.resolution
            )
            print("read {} cameras".format(len(camera_list)))
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
        # filter out cameras, whose video doesnt exist.
        # self.video_name_list_list,
        loaded_video_info, camera_list, format_msg = self._parse_video_names(
            camera_list, self.video_dir
        )
        if format_msg == "FormatInVideo":
            self.video_name_list_list = loaded_video_info

        elif format_msg == "FormatInImage":
            self.video_numpy_list = loaded_video_info
            self.video_name_list_list = None

        if "num_frames" not in meta_dict:
            meta_dict["num_frames"] = len(camera_list)

        self.meta_dict = meta_dict
        self.camera_list = camera_list
        self.test_camera_list = test_camera_list

        self._num_frames = meta_dict[
            "num_frames"
        ]  # number of timestamps in the dataset
        self.num_cameras = meta_dict["num_cameras"]
        self.dataset_len = len(self.camera_list)

        self._preload_imgs(self.camera_list)

    def _preload_imgs(self, camera_list):
        np_uint8_rgba_list = []
        print("preloading {} images...".format(len(camera_list)))
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

    def _parse_video_names(self, camera_list, video_dir):
        video_name_list_list = []

        video_names = [_ for _ in os.listdir(video_dir) if _.endswith(".mp4")]

        if len(video_names) > 0:
            ret_camera_list = []
            for cam in tqdm(camera_list):
                img_name = os.path.basename(cam.img_path).split(".")[0]
                ret_list = [
                    _
                    for _ in video_names
                    if _.startswith(img_name) and _[:4] == img_name[:4]
                ]
                if len(ret_list) > 0:
                    print("video found for camera: ", cam.img_path, ret_list)
                    video_name_list_list.append(ret_list)
                    ret_camera_list.append(cam)
            return video_name_list_list, ret_camera_list, "FormatInVideo"
        else:
            # no video found, concate all images inside "video_dir" to a video
            # corresponding camera is: viewpoint_name = os.path.dirname(video_dir).split("/")[-1]
            ret_video_list = []
            img_names = [_ for _ in os.listdir(video_dir) if _.endswith(".png")]
            assert len(img_names) > 0, "no images or videos found in video_dir!"
            img_names = sorted(img_names)

            img_names = img_names[:60] + img_names[-5:]

            viewpoint_name = os.path.dirname(video_dir).split("/")[-1]
            ret_camera_list = []
            for cam in tqdm(camera_list):
                img_name = os.path.basename(cam.img_path).split(".")[0]
                # print(img_name, viewpoint_name)
                if viewpoint_name.startswith(img_name):
                    ret_camera_list.append(cam)

            assert (
                len(ret_camera_list) > 0
            ), "no camera found for video!, viewpoint_name: {}".format(viewpoint_name)
            # load imgs:
            _img_list = []
            for img_name in tqdm(img_names):
                img_path = os.path.join(video_dir, img_name)
                im_data = read_uint8_rgba(img_path, self.resolution)[..., :3]
                _img_list.append(im_data[np.newaxis, ...])
            video_numpy = np.concatenate(_img_list, axis=0)
            ret_video_list.append(video_numpy)

            return ret_video_list, ret_camera_list, "FormatInImage"

    def __len__(self):
        return self.dataset_len

    @property
    def num_frames(self):
        return self._num_frames

    def __getitem__(self, idx):
        cam = self.camera_list[idx]
        rgba = self.np_uint8_rgba_list[idx]

        if self.video_name_list_list is not None:
            video_names_list = self.video_name_list_list[idx]
            _rand_ind = random.randint(0, len(video_names_list) - 1)

            video_path = os.path.join(self.video_dir, video_names_list[_rand_ind])
            video_reader = VideoReader(video_path)
            video_length = len(video_reader)
            video_clip = (
                torch.from_numpy(
                    video_reader.get_batch(np.arange(0, video_length)).asnumpy()
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )  # [nf, 3, H, W], RGB, [0, 255]
        else:
            video_clip = (
                torch.from_numpy(self.video_numpy_list[idx])
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        video_clip = video_clip / 255.0
        # video_clip = self.video_transforms(video_clip)

        norm_data = rgba / 255.0

        img = norm_data[:, :, :3] * norm_data[:, :, 3:4]

        if idx > len(self.mask_float32_list):
            mask = norm_data[:, :, 3:4]
        else:
            mask = self.mask_float32_list[idx]

        mask = torch.from_numpy(mask.astype(np.float32)).permute(2, 0, 1)
        # shape convert from HWC to CHW
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)

        # add read flow, depth later.
        ret_dict = {
            "img": img,  # [H, W, 3] value in [0, 1]
            "mask": mask,  # [H, W, 1], 1 for foreground, 0 for background
            "cam": cam,
            "video_clip": video_clip,  # [nf, 3, H, W] value in [0, 1]
            "idx": torch.tensor([idx]).long(),
        }

        return ret_dict

    def _read_camera_transforms(
        self,
        camera_transform_file,
        img_hw=None,
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
    ):
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


def camera_dataset_collate_fn(batch):
    ret_dict = {
        "cam": [],
    }

    for key in batch[0].keys():
        if key == "cam":
            ret_dict[key].extend([item[key] for item in batch])
        elif key == "timestamp":
            ret_dict[key] = torch.tensor([item[key] for item in batch])
        else:
            ret_dict[key] = torch.stack([item[key] for item in batch], dim=0)

    return ret_dict


if __name__ == "__main__":
    # dataset_dir = "data/multiview/dragon/merged"
    # dataset_dir = "/tmp/tmp_tyz_data/ficus/"

    dataset_dir = "../../../../../dataset/3D_capture/purple_branches_colmap"
    dataset_dir = "../../../../../dataset/physics_dreamer/llff_flower_undistorted"

    dataset = MultiviewVideoDataset(
        dataset_dir,
    )

    data = dataset[0]

    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            print(key, val.shape)
        else:
            print(key, type(val))
