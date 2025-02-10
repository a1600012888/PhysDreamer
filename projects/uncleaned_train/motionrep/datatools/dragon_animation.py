import bpy

# Clear existing data
bpy.ops.wm.read_factory_settings(use_empty=True)

# 1. Import the FBX file
fbx_path = "/local/cg/rundi/data/New-FBX-BVH_Z-OO/Truebone_Z-OO/Dragon/Wyvern-Fly.fbx"
# fbx_path = "../../../data/motion_dataset/pirate-flag-animated/source/pirate_flag.fbx"
bpy.ops.import_scene.fbx(filepath=fbx_path)


# 2. Set up the camera and lighting (assuming they aren't in the FBX)
# Add a camera
bpy.ops.object.camera_add(location=(0, -14, 7))
camera = bpy.context.object
camera.rotation_euler = (1.5708, 0, 0)  # Point the camera towards the origin

# Ensure the camera is in the scene and set it as the active camera
if "Camera" in bpy.data.objects:
    bpy.context.scene.camera = camera
else:
    print("Camera not added!")

# Add a light
bpy.ops.object.light_add(type="SUN", location=(15, -15, 15))

# 3. Set up the render settings
bpy.context.scene.render.engine = "CYCLES"  # or 'EEVEE'
bpy.context.scene.render.image_settings.file_format = "FFMPEG"
bpy.context.scene.render.ffmpeg.format = "MPEG4"
bpy.context.scene.render.ffmpeg.codec = "H264"
bpy.context.scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
bpy.context.scene.render.filepath = ".data/dragon/dragon.mp4"
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 250  # Adjust this based on your needs

# 4. Render the animation
bpy.ops.render.render(animation=True)
