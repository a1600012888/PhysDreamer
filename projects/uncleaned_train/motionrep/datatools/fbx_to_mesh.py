import bpy
import os
import sys


def convert_to_mesh(fbx_path, output_dir):
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # Assuming the imported object is the active object
    obj = bpy.context.active_object
    for obj in bpy.context.selected_objects:
        print("obj: ", obj.name, obj.type)
    mesh_objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]

    # Get the active object (assuming it's the imported FBX mesh)
    obj = bpy.context.active_object

    for obj in mesh_objects:
        # Add the subdivision modifier
        mod = obj.modifiers.new(name="Subdivision", type="SUBSURF")
        mod.levels = 1  # Set this to the desired subdivision level
        mod.render_levels = 2  # Set this to the desired subdivision level for rendering

        # Apply the modifier
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # Set the start and end frames (modify these values if needed)
    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end

    # Iterate through each frame and export
    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)

        # Construct the filename
        filename = os.path.join(
            output_dir, f"frame_{frame:04}.obj"
        )  # Change to .blend for Blender format

        # Export to OBJ format
        bpy.ops.export_scene.obj(filepath=filename, use_selection=True)

        # Uncomment the line below and comment the above line if you want to export in Blender format
        # bpy.ops.wm.save_as_mainfile(filepath=filename, copy=True)

    print("Export complete!")


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

    convert_to_mesh(inp_fpx_path, output_dir)
    convert_obj_to_traj(output_dir)


if __name__ == "__main__":
    main()
