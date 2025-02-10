import bpy
import numpy as np
import sys
import point_cloud_utils as pcu
import os


def convert(fbx_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # 1. Import the FBX file
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    print("loaded fbx from: ", fbx_path)

    # Assuming the FBX file has one main mesh object, get it
    mesh_obj = bpy.context.selected_objects[1]

    # 2. Duplicate the mesh for the first frame
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    bpy.ops.object.duplicate()
    static_mesh = bpy.context.object

    # Apply the first frame's pose to the static mesh
    bpy.context.scene.frame_set(1)
    bpy.ops.object.modifier_apply({"object": static_mesh}, modifier="Armature")

    # 3. Calculate and store vertex offsets for each subsequent frame
    vertex_traj_list = []
    num_frames = bpy.context.scene.frame_end

    for frame in range(1, num_frames + 1):
        bpy.context.scene.frame_set(frame)

        bpy.context.view_layer.update()
        # Update the mesh to the current frame's pose
        # mesh_obj.data.update()

        all_pts_3d = []
        for v1, v2 in zip(static_mesh.data.vertices, mesh_obj.data.vertices):
            pts_3d = v2.co
            all_pts_3d.append(pts_3d)

        vertex_traj_list.append(np.array(all_pts_3d))

    vertex_traj_list = np.stack(vertex_traj_list, axis=0)

    # Now, frame_offsets contains the vertex offsets for each frame
    vertex_array = vertex_traj_list[0]  # first frame

    # get face indx
    bpy.context.view_layer.objects.active = static_mesh
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    faces_list = [list(face.vertices) for face in static_mesh.data.polygons]

    faces_array = np.array(faces_list, dtype=np.int32)
    vertices = np.array([v.co for v in static_mesh.data.vertices])
    print("vertices shape: ", vertices.shape)

    print(
        "num_frames: ",
        num_frames,
        "offsets shape",
        vertex_traj_list.shape,
        "num_faces",
        faces_array.shape,
        "max offset",
        np.max(vertex_traj_list - vertex_array[np.newaxis, :, :]),
        np.min(vertex_traj_list - vertex_array[np.newaxis, :, :]),
    )

    mean = np.mean(vertices, axis=0)
    max_range = np.max(np.max(vertices, axis=0) - np.min(vertices, axis=0))
    print("max_range: ", max_range, "mean: ", mean)

    # normalize
    # vertex_array = (vertex_array - mean[np.newaxis, :]) / max_range
    # vertex_traj_list = (vertex_traj_list - mean[np.newaxis, np.newaxis, :]) / max_range

    return faces_array, vertex_array, vertex_traj_list


def convert2(fbx_path):
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # Assuming the imported object is the active object
    obj = bpy.context.active_object
    for obj in bpy.context.selected_objects:
        print("obj: ", obj.name, obj.type)
    mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]

    # Ensure it's in object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    # Get the total number of frames in the scene
    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end

    # Create a dictionary to store vertex positions for each frame
    vertex_data_list = []

    # Get the dependency graph
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Iterate over each frame
    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)

        # Update the scene to reflect changes
        bpy.context.view_layer.update()

        ret_list = []
        for mesh_obj in mesh_objects:
            # deformed_mesh = mesh_obj.to_mesh()
            # Extract vertex positions for the current frame
            # vertex_positions = [vertex.co for vertex in deformed_mesh.vertices]
            # vertex_positions = [vertex.co.copy() for vertex in deformed_mesh.vertices]

            duplicated_obj = mesh_obj.copy()
            duplicated_obj.data = mesh_obj.data.copy()
            bpy.context.collection.objects.link(duplicated_obj)

            # Make the duplicated object the active object
            bpy.context.view_layer.objects.active = duplicated_obj
            duplicated_obj.select_set(True)
            print("duplicated_obj.modifiers", duplicated_obj.modifiers)

            for mod in duplicated_obj.modifiers:
                bpy.ops.object.modifier_apply(
                    {"object": duplicated_obj}, modifier=mod.name
                )

            # if "Armature" in duplicated_obj.modifiers:
            #     bpy.ops.object.modifier_apply(
            #         {"object": duplicated_obj}, modifier="Armature"
            #     )

            # Extract vertex positions from the duplicated object
            vertex_positions = [
                vertex.co.copy() for vertex in duplicated_obj.data.vertices
            ]

            ret_list += vertex_positions

        # Convert to numpy array and store in the dictionary
        vertex_data_list.append(np.array(ret_list))

    vertex_traj_list = np.stack(vertex_data_list, axis=0)
    print(
        "offsets shape",
        vertex_traj_list.shape,
        "max offset",
        np.max(vertex_traj_list - vertex_traj_list[0:1, :, :]),
        np.min(vertex_traj_list - vertex_traj_list[0:1, :, :]),
    )

    # bpy.ops.object.mode_set(mode="EDIT")
    # bpy.ops.mesh.select_all(action="SELECT")
    # bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    # bpy.ops.object.mode_set(mode="OBJECT")
    if bpy.context.active_object.type == "MESH":
        obj = bpy.context.active_object

        # Set the mode to 'EDIT'
        bpy.ops.object.mode_set(mode="EDIT")

        # Ensure the mesh is the active object and is in edit mode
        if bpy.context.mode == "EDIT_MESH" and bpy.context.object == obj:
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.quads_convert_to_tris(
                quad_method="BEAUTY", ngon_method="BEAUTY"
            )
            bpy.ops.object.mode_set(mode="OBJECT")
        else:
            print("Failed to set the correct context.")
    else:
        print("Active object is not a mesh.")
    faces_list = []
    for mesh_obj in mesh_objects:
        _fl = [list(face.vertices) for face in obj.data.polygons]
        faces_list += _fl

    faces_array = np.array(faces_list, dtype=np.int32)
    vertex_array = vertex_traj_list[0]  # first frame
    print("face shape", faces_array.shape)
    return faces_array, vertex_array, vertex_traj_list


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :]  # get all args after "--"
    print(argv)
    fbx_path = argv[0]  # input mesh path
    output_dir = argv[1]  # output dir
    # num_frames = int(argv[2])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    faces_array, vertex_array, vertex_traj_array = convert2(fbx_path)

    save_mesh_path = os.path.join(output_dir, "mesh0.obj")
    pcu.save_mesh_vf(save_mesh_path, vertex_array, faces_array)

    save_traj_path = os.path.join(output_dir, "traj.npy")
    np.save(save_traj_path, vertex_traj_array)


if __name__ == "__main__":
    main()
