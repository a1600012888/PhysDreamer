import bpy
import os
import numpy as np
import math
import sys
import struct
import collections
from mathutils import Matrix, Quaternion, Vector
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


def render_scene(camera, output_path, mask_name="U3DMesh"):
    bpy.context.scene.render.film_transparent = True

    setup_alpha_mask(mask_name, 1)
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
    light_1.location = (20, 20, 20)  # Adjust location as needed
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
    light_2.location = (20, -20, 20)  # Adjust location as needed
    light_2.rotation_euler = (
        np.radians(45),
        np.radians(180),
        np.radians(45),
    )  # Adjust rotation for direction

    # Add second directional light (Sun lamp)
    light_data_3 = bpy.data.lights.new(name="Directional_Light_3", type="SUN")
    light_data_3.energy = 3  # Adjust energy as needed
    light_3 = bpy.data.objects.new(name="Directional_Light_3", object_data=light_data_2)
    bpy.context.collection.objects.link(light_3)
    light_3.location = (-20, 20, 20)  # Adjust location as needed
    light_3.rotation_euler = (
        np.radians(-135),
        np.radians(0),
        np.radians(45),
    )  # Adjust rotation for direction


def create_mesh_from_obj(obj_file_path):
    # Clear existing mesh objects in the scene
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()

    bpy.ops.import_scene.obj(filepath=obj_file_path)

    # Assuming the imported object is the active object
    obj = bpy.context.active_object
    num_obj = 0
    for obj in bpy.context.selected_objects:
        print("obj mesh name: ", obj.name, obj.type)
        num_obj += 1
    if num_obj > 2:
        raise ValueError("More than one object in the scene.")
    mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]

    return obj.name, mesh_objects


def get_focal_length(camera_obj, render_resolution):
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

    return fx, fy


def normalize_mesh(transform_meta_path, mesh_objects):
    import json

    if os.path.exists(transform_meta_path):
        with open(transform_meta_path, "r") as f:
            meta_dict = json.load(f)
    # obj = bpy.context.active_object

    for obj in mesh_objects:
        # Ensure the object is in object mode
        # bpy.ops.object.mode_set(mode="OBJECT")

        scale_ = 1.0 / meta_dict["scale"]
        center = Vector(meta_dict["center"])
        # Apply the scale
        print("old scale: ", obj.scale)
        # obj.location -= center
        obj.scale *= scale_


def apply_rotation(mesh_objects):
    for obj in mesh_objects:
        R_np = [[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]]
        R_blender = Matrix(R_np).transposed()

        # Convert the rotation matrix to a quaternion
        quaternion = R_blender.to_quaternion()

        # Set the active object's rotation to this quaternion
        print("rotation", quaternion, obj.rotation_quaternion)
        obj.rotation_quaternion = obj.rotation_quaternion @ quaternion


def get_textures(
    texture_dir="/local/cg/rundi/data/motion_dataset/pirate-flag-animated/source/textures",
):
    # Ensure the "flag" object is selected
    # bpy.context.view_layer.objects.active = bpy.data.objects["flag"]

    obj = bpy.data.objects["flag.001_Plane.001"]

    # Create a new material or get the existing one
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="FBX_Material")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]

    # Use nodes for the material
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add a Principled BSDF shader and connect it to the Material Output
    shader = nodes.new(type="ShaderNodeBsdfPrincipled")
    shader.location = (0, 0)

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (400, 0)
    mat.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    # Load textures and create the corresponding nodes

    textures = {
        "Base Color": os.path.join(texture_dir, "pirate_flag_albedo.jpg"),
        "Metallic": os.path.join(texture_dir, "pirate_flag_metallic.jpg"),
        "Normal": os.path.join(texture_dir, "pirate_flag_normal.png"),
        "Roughness": os.path.join(texture_dir, "pirate_flag_roughness.jpg"),
    }
    # ... [rest of the script]

    ao_texture = nodes.new(type="ShaderNodeTexImage")
    ao_texture.location = (-400, -200)
    ao_texture.image = bpy.data.images.load(
        filepath=os.path.join(texture_dir, "pirate_flag_AO.jpg")
    )  # Adjust filepath if needed

    mix_rgb = nodes.new(type="ShaderNodeMixRGB")
    mix_rgb.location = (-200, 0)
    mix_rgb.blend_type = "MULTIPLY"
    mix_rgb.inputs[
        0
    ].default_value = 1.0  # Factor to 1 to fully use the multiply operation

    mat.node_tree.links.new(ao_texture.outputs["Color"], mix_rgb.inputs[2])

    for i, (input_name, filename) in enumerate(textures.items()):
        tex_image = nodes.new(type="ShaderNodeTexImage")
        tex_image.location = (-400, i * 200)
        tex_image.image = bpy.data.images.load(
            filepath=filename
        )  # Adjust filepath if needed

        if input_name == "Base Color":
            mat.node_tree.links.new(tex_image.outputs["Color"], mix_rgb.inputs[1])
            mat.node_tree.links.new(mix_rgb.outputs["Color"], shader.inputs[input_name])
        elif input_name == "Normal":
            normal_map_node = nodes.new(type="ShaderNodeNormalMap")
            normal_map_node.location = (-200, i * 200)
            mat.node_tree.links.new(
                tex_image.outputs["Color"], normal_map_node.inputs["Color"]
            )
            mat.node_tree.links.new(
                normal_map_node.outputs["Normal"], shader.inputs["Normal"]
            )
        else:
            mat.node_tree.links.new(
                tex_image.outputs["Color"], shader.inputs[input_name]
            )


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :]  # get all args after "--"
    print(argv)
    inp_fpx_path = argv[0]  # input mesh path
    output_dir = argv[1]  # output dir
    num_views = int(argv[2])
    radius = 3

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_output_dir = os.path.join(output_dir, "images")
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)

    transform_meta_path = os.path.join(os.path.dirname(inp_fpx_path), "meta.json")

    # Create the 3D mesh in Blender from your data. no normalize
    my_mesh_name, mesh_objects = create_mesh_from_obj(inp_fpx_path)

    normalize_mesh(transform_meta_path, mesh_objects)
    # apply_rotation(mesh_objects)

    get_textures()

    object_center = Vector((0.0, 0.0, 0.5))

    print("look at object center: ", object_center)

    setup_light()
    # Set up rendering parameters
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.resolution_x = 1080
    bpy.context.scene.render.resolution_y = 720

    # bpy.context.scene.render.resolution_x = 720
    # bpy.context.scene.render.resolution_y = 480

    camera = create_camera((1, 1, 1), (0, 0, 0))
    fx, fy = get_focal_length(
        camera,
        (bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y),
    )

    transform_dict = {
        "frames": [],
        "camera_angle_x": focal2fov(fx, bpy.context.scene.render.resolution_x),
    }
    img_indx = 0
    num_elevations = 6
    for j in range(num_elevations):
        num_imgs = num_views // num_elevations
        for i in range(num_imgs):
            angle = 2 * math.pi * i / num_imgs + math.pi / 6.0
            x = object_center.x + radius * math.cos(angle)
            y = object_center.y + radius * math.sin(angle)
            z = (
                object_center.z + (j - num_elevations / 2.0) * radius / num_elevations
            )  # Adjust this if you want the camera to be above or below the object's center

            camera = create_camera((x, y, z), (0, 0, 0))
            rot_quant = set_camera_look_at(camera, object_center)
            bpy.context.view_layer.update()

            # also prepare transforms.json
            fname = f"images/img_{img_indx}"
            cam2world = np.array(camera.matrix_world)
            transform_dict["frames"].append(
                {"file_path": fname, "transform_matrix": cam2world.tolist()}
            )

            render_scene(
                camera,
                os.path.join(img_output_dir, f"img_{img_indx}.png"),
                my_mesh_name,
            )
            img_indx += 1

    trans_fpath = os.path.join(output_dir, "transforms_train.json")
    import json

    with open(trans_fpath, "w") as f:
        json.dump(transform_dict, f)

    transform_dict["frames"] = transform_dict["frames"][::4]
    trans_fpath = os.path.join(output_dir, "transforms_test.json")

    with open(trans_fpath, "w") as f:
        json.dump(transform_dict, f)


if __name__ == "__main__":
    main()
    # find_material()
