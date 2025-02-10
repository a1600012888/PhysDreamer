import cv2
import imageio
import numpy as np
import mediapy
import os
import PIL


def read_video_cv2(video_path, rgb=True):
    """Read video using cv2, return [T, 3, H, W] array, fps"""

    # BGR
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret_list = []
    for i in range(num_frame):
        ret, frame = cap.read()
        if ret:
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, [2, 0, 1])  # [3, H, W]
            ret_list.append(frame[np.newaxis, ...])
        else:
            break
    cap.release()
    ret_array = np.concatenate(ret_list, axis=0)  # [T, 3, H, W]
    return ret_array, fps


def save_video_cv2(video_path, img_list, fps):
    # BGR

    if len(img_list) == 0:
        return
    h, w = img_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )  # cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for frame in img_list:
        writer.write(frame)
    writer.release()


def save_video_imageio(video_path, img_list, fps):
    """
    Img_list: [[H, W, 3]]
    """
    if len(img_list) == 0:
        return
    writer = imageio.get_writer(video_path, fps=fps)
    for frame in img_list:
        writer.append_data(frame)

    writer.close()


def save_gif_imageio(video_path, img_list, fps):
    """
    Img_list: [[H, W, 3]]
    """
    if len(img_list) == 0:
        return
    assert video_path.endswith(".gif")

    imageio.mimsave(video_path, img_list, format="GIF", fps=fps)


def save_video_mediapy(video_frames, output_video_path: str = None, fps: int = 14):
    # video_frames: [N, H, W, 3]
    if isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    mediapy.write_video(output_video_path, video_frames, fps=fps, qp=18)
