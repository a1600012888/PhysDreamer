import os
import numpy as np


def anime_read(filename):
    """
    filename: path of .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: vertice data of the 1st frame (3D positions in x-y-z-order). [nv, 3]
        face_data: riangle face data of the 1st frame. [nt, 3] dtype = int32
        offset_data: 3D offset data from the 2nd to the last frame. [nf, nv, 3]
    """
    f = open(filename, "rb")
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    """check data consistency"""
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data


def extract_trajectory(
    trajectory_array: np.ndarray,
    topk_freq: int = 8,
):
    """
    Args:
        trajectory_array: [nf, nv, 3]. The 3D position of each point in each frame.
        topk_freq: int. FFT frequency.
    """

    # doing fft on trajectory_array
    # [nf, nv, 3]
    trajectory_fft = np.fft.fft(trajectory_array, axis=0)
    # only keep topk_freq
    # [topk_freq, nv, 3]
    trajectory_fft = trajectory_fft[:topk_freq, :, :]
    trajectory_fft[topk_freq:-topk_freq, :, :] = 0

    # doing ifft on trajectory_fft
    # [nf, nv, 3]
    trajectory_array = np.fft.ifft(trajectory_fft, axis=0).real


def main():
    import argparse
    import point_cloud_utils as pcu

    parser = argparse.ArgumentParser(description="None description")

    parser.add_argument("--input", type=str, help="input path")
    parser.add_argument("--output_dir", type=str, help="output path")
    parser.add_argument(
        "--skip",
        type=int,
        default=-1,
        help="skipping between frame saving. -1 indicates only save first frame",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    inp_ani_path = args.input

    nf, nv, nt, vert_data, face_data, offset_data = anime_read(inp_ani_path)
    #  face_data:  offset_data [nf, nv, 3]

    # normalize
    verices_center = np.mean(vert_data, axis=0)
    max_range = np.max(np.max(vert_data, axis=0) - np.min(vert_data, axis=0))
    print(
        max_range.shape,
        max_range,
        verices_center.shape,
        verices_center,
        vert_data.shape,
    )
    vert_data = (vert_data - verices_center[np.newaxis, :]) / max_range
    offset_data = offset_data / max_range

    # save trajectory as numpy array

    # [nf, nv, 3]
    trajectory_array = offset_data + vert_data[None, :, :]
    trajectory_array = np.concatenate([vert_data[None, :, :], trajectory_array], axis=0)
    out_traj_path = os.path.join(args.output_dir, "trajectory.npy")
    print("trajectory array of shape [nf, nv, 3]. key: data", trajectory_array.shape)
    save_dict = {
        "help": "trajectory array of shape [nf, nv, 3]. key: data",
        "data": trajectory_array,
    }

    # np.savez(out_traj_path, save_dict)
    # np.save(out_traj_path, trajectory_array)

    if args.skip == -1:
        # save mesh as .obj
        out_obj_path = os.path.join(args.output_dir, "mesh0.obj")
        pcu.save_mesh_vf(out_obj_path, vert_data, face_data)

        return

    for i in range(nf):
        if i % args.skip != 0:
            continue
        out_obj_path = os.path.join(args.output_dir, "mesh{}.obj".format(i))
        pcu.save_mesh_vf(out_obj_path, trajectory_array[i], face_data)


if __name__ == "__main__":
    main()
