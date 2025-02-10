import bpy
import os
import sys


def convert_to_mesh(fbx_path, output_dir):
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # Assuming the imported object is the active object
    original_obj = bpy.context.active_object

    for obj in bpy.context.selected_objects:
        print("obj: ", obj.name, obj.type)

    # Duplicate the original object
    bpy.ops.object.duplicate()
    duplicate_obj = bpy.context.active_object

    # Remove shape keys from the duplicate
    if duplicate_obj.data.shape_keys:
        bpy.context.view_layer.objects.active = duplicate_obj
        bpy.ops.object.shape_key_remove(all=True)

    # Add and apply the subdivision modifier to the duplicate
    mod = duplicate_obj.modifiers.new(name="Subdivision", type="SUBSURF")
    mod.levels = 1
    mod.render_levels = 1
    bpy.context.view_layer.objects.active = duplicate_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    # Set the start and end frames
    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end

    # Iterate through each frame, set the shape of the duplicate to match the original, and export
    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)

        # Transfer shape from original to duplicate (this assumes the original animation uses shape keys)
        if original_obj.data.shape_keys:
            for key_block in original_obj.data.shape_keys.key_blocks:
                duplicate_obj.data.vertices.foreach_set("co", key_block.data[:])

        # Construct the filename
        filename = os.path.join(output_dir, f"frame_{frame:04}.obj")

        # Export to OBJ format
        bpy.ops.export_scene.obj(filepath=filename, use_selection=True)

    print("Export complete!")


def subdivde_mesh(mesh_directory, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List all files in the directory
    all_files = os.listdir(mesh_directory)

    # Filter for .obj files (or change to the format you're using)
    obj_files = [f for f in all_files if f.endswith(".obj")]

    for obj_file in obj_files:
        # Construct full path
        full_path = os.path.join(mesh_directory, obj_file)

        # Clear existing mesh data
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="MESH")
        bpy.ops.object.delete()

        # Import the mesh
        bpy.ops.import_scene.obj(filepath=full_path)

        # Select all imported objects (assuming they are meshes)
        bpy.ops.object.select_all(action="SELECT")

        # Apply subdivision
        for obj in bpy.context.selected_objects:
            if obj.type == "MESH":
                print("apply subdivide to: ", obj.name)
                mod = obj.modifiers.new(name="Subdivision", type="SUBSURF")
                mod.levels = 1
                mod.render_levels = 1
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_apply(modifier=mod.name)

        # Export the mesh with subdivision
        output_path = os.path.join(output_directory, obj_file)
        bpy.ops.export_scene.obj(filepath=output_path, use_selection=True)

    print("Processing complete!")


def convert_obj_to_traj(meshes_dir):
    import glob
    import numpy as np
    import point_cloud_utils as pcu

    meshes = sorted(glob.glob(os.path.join(meshes_dir, "*.obj")))
    print("total of {} meshes: ".format(len(meshes)), meshes[:5], "....")
    traj = []
    R_mat = np.array(
        [[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]],
    )
    for mesh in meshes:
        print(mesh)
        verts, faces = pcu.load_mesh_vf(mesh)
        verts = R_mat[np.newaxis, :, :] @ verts[:, :, np.newaxis]
        verts = verts.squeeze(axis=-1)
        traj.append(verts)
    traj = np.array(traj)

    print("final traj shape", traj.shape)

    save_path = os.path.join(meshes_dir, "traj.npy")
    np.save(save_path, traj)

    save_path = os.path.join(meshes_dir, "../", "traj.npy")
    np.save(save_path, traj)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :]  # get all args after "--"
    print(argv)
    inp_fpx_path = argv[0]  # input mesh path
    output_dir = argv[1]  # output dir
    # num_frames = int(argv[2])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join(output_dir, "meshes")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # convert_to_mesh(inp_fpx_path, output_dir)
    dense_output_dir = os.path.join(output_dir, "denser_mesh")
    os.makedirs(dense_output_dir, exist_ok=True)
    subdivde_mesh(output_dir, dense_output_dir)
    convert_obj_to_traj(dense_output_dir)


if __name__ == "__main__":
    main()
