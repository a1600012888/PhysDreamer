import bpy
import os
import numpy as np
import math
import sys
import struct
import collections
from mathutils import Matrix, Quaternion
from scipy.spatial.transform import Rotation


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def create_camera(location, rotation):
    # Create a new camera
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    return bpy.context.active_object


def set_camera_look_at(camera, target_point):
    # Compute the direction vector from the camera to the target point
    direction = target_point - camera.location
    # Compute the rotation matrix to align the camera's -Z axis to this direction
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return rot_quat


def setup_alpha_mask(obj_name, pass_index=1):
    # Set the object's pass index
    obj = bpy.data.objects[obj_name]
    obj.pass_index = pass_index

    # Enable the Object Index pass for the active render layer
    bpy.context.view_layer.use_pass_object_index = True

    # Enable 'Use Nodes':
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Add Render Layers node
    render_layers = tree.nodes.new("CompositorNodeRLayers")

    # Add Composite node (output)
    composite = tree.nodes.new("CompositorNodeComposite")

    # Add ID Mask node
    id_mask = tree.nodes.new("CompositorNodeIDMask")
    id_mask.index = pass_index

    # Add Set Alpha node
    set_alpha = tree.nodes.new("CompositorNodeSetAlpha")

    # Connect nodes
    tree.links.new(render_layers.outputs["Image"], set_alpha.inputs["Image"])
    tree.links.new(render_layers.outputs["IndexOB"], id_mask.inputs[0])
    tree.links.new(id_mask.outputs[0], set_alpha.inputs["Alpha"])
    tree.links.new(set_alpha.outputs["Image"], composite.inputs["Image"])


def render_scene(camera, output_path):
    bpy.context.scene.render.film_transparent = True

    setup_alpha_mask("MyMeshObject", 1)
    # Set the active camera
    bpy.context.scene.render.image_settings.color_mode = "RGBA"

    bpy.context.scene.camera = camera

    # Set the output path for the render
    bpy.context.scene.render.filepath = output_path

    # Render the scene
    bpy.ops.render.render(write_still=True)


def setup_light():
    # Add first directional light (Sun lamp)
    light_data_1 = bpy.data.lights.new(name="Directional_Light_1", type="SUN")
    light_data_1.energy = 3  # Adjust energy as needed
    light_1 = bpy.data.objects.new(name="Directional_Light_1", object_data=light_data_1)
    bpy.context.collection.objects.link(light_1)
    light_1.location = (10, 10, 10)  # Adjust location as needed
    light_1.rotation_euler = (
        np.radians(45),
        np.radians(0),
        np.radians(45),
    )  # Adjust rotation for direction

    # Add second directional light (Sun lamp)
    light_data_2 = bpy.data.lights.new(name="Directional_Light_2", type="SUN")
    light_data_2.energy = 5  # Adjust energy as needed
    light_2 = bpy.data.objects.new(name="Directional_Light_2", object_data=light_data_2)
    bpy.context.collection.objects.link(light_2)
    light_2.location = (10, -10, 10)  # Adjust location as needed
    light_2.rotation_euler = (
        np.radians(45),
        np.radians(180),
        np.radians(45),
    )  # Adjust rotation for direction


def create_mesh_from_data(vertices, faces):
    # Clear existing mesh objects in the scene
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()

    vertices_list = vertices.tolist()
    faces_list = faces.tolist()

    # Create a new mesh
    mesh_name = "MyMesh"
    mesh = bpy.data.meshes.new(name=mesh_name)
    obj = bpy.data.objects.new("MyMeshObject", mesh)

    # Link it to the scene
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Load the mesh data
    mesh.from_pydata(vertices_list, [], faces_list)
    mesh.update()

    # mesh_data = bpy.data.meshes.new(mesh_name)
    # mesh_data.from_pydata(vertices_list, [], faces_list)
    # mesh_data.update()
    # the_mesh = bpy.data.objects.new(mesh_name, mesh_data)
    # the_mesh.data.vertex_colors.new()  # init color
    # bpy.context.collection.objects.link(the_mesh)

    # UV unwrap the mesh
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")

    # Texture the mesh based on its normals
    mat = bpy.data.materials.new(name="NormalMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    normal_node = mat.node_tree.nodes.new(type="ShaderNodeNormal")
    geometry = mat.node_tree.nodes.new(type="ShaderNodeNewGeometry")

    # mat.node_tree.links.new(geometry.outputs["Normal"], normal_node.inputs["Normal"])
    # mat.node_tree.links.new(normal_node.outputs["Dot"], bsdf.inputs["Base Color"])
    mat.node_tree.links.new(geometry.outputs["Normal"], bsdf.inputs["Base Color"])

    obj.data.materials.append(mat)

    return None


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


def write_images_binary(images, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def get_colmap_camera(camera_obj, render_resolution):
    """
    Extract the intrinsic matrix from a Blender camera.

    Args:
    - camera_obj: The Blender camera object.
    - render_resolution: Tuple of (width, height) indicating the render resolution.

    Returns:
    - colmap_camera: dict of ["id", "model", "width", "height", "params"]
    """

    # Get the camera data
    cam = camera_obj.data

    # Ensure it's a perspective camera
    if cam.type != "PERSP":
        raise ValueError("Only 'PERSP' camera type is supported.")

    # Image resolution
    width, height = render_resolution

    # Sensor width and height in millimeters
    sensor_width_mm = cam.sensor_width
    sensor_height_mm = cam.sensor_height

    # Calculate the focal length in pixels
    fx = (cam.lens / sensor_width_mm) * width
    fy = (cam.lens / sensor_height_mm) * height

    # Principal point, usually at the center of the image
    cx = width / 2.0
    cy = height / 2.0

    _cam_dict = {
        "id": 0,
        "model": "PINHOLE",  # PINHOLE
        "width": width,
        "height": height,
        "params": [fx, fy, cx, cy],
    }

    colmap_cameras = {0: Camera(**_cam_dict)}

    print("focal", fx, fy, cx, cy)

    return colmap_cameras


def main():
    import point_cloud_utils as pcu

    argv = sys.argv
    argv = argv[argv.index("--") + 1 :]  # get all args after "--"
    print(argv)
    inp_mesh_path = argv[0]  # input mesh path
    output_dir = argv[1]  # output dir
    # num_frames = int(argv[2])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_output_dir = os.path.join(output_dir, "images")
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)

    tmp_mesh_path = "data/deer_attack/mesh0.obj"

    vertices, faces = pcu.load_mesh_vf(tmp_mesh_path)
    # normalize
    verices_center = np.mean(vertices, axis=0)
    max_range = np.max(np.max(vertices, axis=0) - np.min(vertices, axis=0))
    print(
        max_range.shape, max_range, verices_center.shape, verices_center, vertices.shape
    )

    vertices, faces = pcu.load_mesh_vf(inp_mesh_path)

    mesh_name = os.path.basename(inp_mesh_path).split(".")[0]

    vertices = (vertices - verices_center[np.newaxis, :]) / max_range

    # Create the 3D mesh in Blender from your data.
    obj = create_mesh_from_data(vertices, faces)

    object_center = bpy.context.scene.objects["MyMeshObject"].location

    # Number of viewpoints
    num_views = 180  # 180
    radius = 6  # Distance of the camera from the object center

    setup_light()
    # Set up rendering parameters
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.resolution_x = 1080
    bpy.context.scene.render.resolution_y = 720

    camera = create_camera((1, 1, 1), (0, 0, 0))
    colmap_camera_dict = get_colmap_camera(
        camera,
        (bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y),
    )

    transform_dict = {
        "frames": [],
        "camera_angle_x": focal2fov(
            colmap_camera_dict[0].params[0], colmap_camera_dict[0].width
        ),
    }
    img_indx = 0
    num_elevations = 6
    colmap_images_dict = {}
    for j in range(num_elevations):
        num_imgs = num_views // num_elevations
        for i in range(num_imgs):
            angle = 2 * math.pi * i / num_imgs
            x = object_center.x + radius * math.cos(angle)
            y = object_center.y + radius * math.sin(angle)
            z = (
                object_center.z + (j - num_elevations / 3.0) * 4.0 / num_elevations
            )  # Adjust this if you want the camera to be above or below the object's center

            camera = create_camera((x, y, z), (0, 0, 0))
            rot_quant = set_camera_look_at(camera, object_center)
            tvec = np.array([x, y, z])
            bpy.context.view_layer.update()

            # plan-1
            # w2c = np.array(camera.matrix_world.inverted())
            # w2c[1:3, :] *= -1.0
            # rotation_matrix = w2c[:3, :3]
            # tvec = w2c[:3, 3]
            # plan-1 end

            # plan-2
            camera_to_world_matrix = camera.matrix_world
            # [4, 4]
            camera_to_world_matrix = np.array(camera_to_world_matrix).copy()
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            camera_to_world_matrix[:3, 1:3] *= -1.0
            w2c = np.linalg.inv(camera_to_world_matrix)
            rotation_matrix = w2c[:3, :3]
            tvec = w2c[:3, 3]

            # c2w rotation
            # rotation_matrix = rot_quant.to_matrix()  # .to_4x4()
            # # w2c rotation
            # rotation_matrix = np.array(rotation_matrix)
            # # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # rotation_matrix[:3, 1:3] *= -1.0
            # rotation_matrix = rotation_matrix.transpose()
            # tvec = (rotation_matrix @ tvec[:, np.newaxis]).squeeze(axis=-1) * -1.0

            rot_quant = Rotation.from_matrix(rotation_matrix).as_quat()
            # print("r shape", rotation_matrix.shape, tvec.shape)

            img_dict = {
                "id": img_indx,
                "qvec": rot_quant,
                "tvec": tvec,
                "camera_id": 0,
                "name": f"img_{img_indx}.png",
                "xys": [[k, k] for k in range(i, i + 10)],  # placeholder
                "point3D_ids": list(range(i, i + 10)),  # placeholder
            }
            colmap_images_dict[img_indx] = BaseImage(**img_dict)

            # also prepare transforms.json
            fname = f"images/img_{img_indx}"
            cam2world = np.array(camera.matrix_world)
            transform_dict["frames"].append(
                {"file_path": fname, "transform_matrix": cam2world.tolist()}
            )

            render_scene(camera, os.path.join(img_output_dir, f"img_{mesh_name}.png"))
            img_indx += 1

            return


if __name__ == "__main__":
    main()
