import mitsuba as mi
import numpy as np
import os
import tqdm
import ffmpeg
import matplotlib.pyplot as plt


mi.set_variant('cuda_ad_rgb')

scene_path = '/home/elec594/Desktop/luigi/Render/mitsuba_gallery/dining-room/scene.xml'
xml_path = scene_path

# X-axis: left 
# Y-axis: up
# Z-axis: backward

out_dir = '/home/elec594/Desktop/luigi/Render/dining/frames_albedo'
os.makedirs(out_dir, exist_ok=True)

num_frames = 50
init_x = -2.75
init_y = 2.0
init_z = -0.25
step_x = -0.01
step_y = 0.0
step_z = -0.01
radius = 0.01

for i in tqdm.tqdm(range(0, num_frames, 5), desc="Rendering frames"):
    x = init_x + step_x * i
    y = init_y + step_y * i
    z = init_z + step_z * i

    rotation_angle = 1.1 * np.pi + (i / num_frames) * 0.7 * np.pi  # Rotate 360 degrees over the sequence
    origin = [x, y, z]
    target = [x + np.cos(rotation_angle) * radius, y , z + np.sin(rotation_angle) * radius]

    scene = mi.load_file(xml_path)
    parameters = mi.traverse(scene)
    parameters['camera.to_world'] = mi.ScalarTransform4f().look_at(
        origin=origin,
        target=target,
        up=[0.0, 1.0, 0.0]
    )
    parameters.update()

    img = mi.render(scene, spp=1024)  # reduce spp for faster preview

    mirror_mask = img[..., -1]
    # mirror_mask = img[..., -1]
    mirror_mask = mirror_mask.numpy()
    mirror_out_dir = os.path.join(out_dir, 'mirror_mask')
    os.makedirs(mirror_out_dir, exist_ok=True)
    np.save(os.path.join(mirror_out_dir, f'frame_{i:03d}_mirror_mask.npy'), mirror_mask)
    # plt.figure()

    # depth = img[...,3:4]
    # depth = depth.numpy()
    # depth_out_dir = os.path.join(out_dir, 'depth')
    # os.makedirs(depth_out_dir, exist_ok=True)
    # np.save(os.path.join(depth_out_dir, f'frame_{i:03d}_depth.npy'), depth)
    # plt.figure()
    # plt.imshow(depth.squeeze(), cmap='plasma')
    # plt.colorbar()
    # plt.title(f'Depth Map - Frame {i:03d}')
    # plt.savefig(os.path.join(depth_out_dir, f'frame_{i:03d}_depth.png'))


scene_path = '/home/elec594/Desktop/luigi/Render/mitsuba_gallery/dining-room/scene_flat.xml'
xml_path = scene_path

# X-axis: left 
# Y-axis: up
# Z-axis: backward

out_dir = '/home/elec594/Desktop/luigi/Render/dining/flat_frames_albedo'
os.makedirs(out_dir, exist_ok=True)


num_frames = 50
init_x = -2.75
init_y = 2.0
init_z = -0.25
step_x = -0.01
step_y = 0.0
step_z = -0.01
radius = 0.01

for i in tqdm.tqdm(range(0, num_frames, 5), desc="Rendering frames"):
    x = init_x + step_x * i
    y = init_y + step_y * i
    z = init_z + step_z * i

    rotation_angle = 1.1 * np.pi + (i / num_frames) * 0.7 * np.pi  # Rotate 360 degrees over the sequence
    origin = [x, y, z]
    target = [x + np.cos(rotation_angle) * radius, y , z + np.sin(rotation_angle) * radius]

    scene = mi.load_file(xml_path)
    parameters = mi.traverse(scene)
    parameters['camera.to_world'] = mi.ScalarTransform4f().look_at(
        origin=origin,
        target=target,
        up=[0.0, 1.0, 0.0]
    )
    parameters.update()

    img = mi.render(scene, spp=1024)  # reduce spp for faster preview
    # mi.util.write_bitmap(os.path.join(out_dir, f'frame_{i:03d}.png'), img)
    # depth = img[...,3:4]
    # depth = depth.numpy()
    # depth_out_dir = os.path.join(out_dir, 'depth')
    # os.makedirs(depth_out_dir, exist_ok=True)
    # np.save(os.path.join(depth_out_dir, f'frame_{i:03d}_depth.npy'), depth)
    # plt.figure()
    # plt.imshow(depth.squeeze(), cmap='plasma')
    # plt.colorbar()
    # plt.title(f'Depth Map - Frame {i:03d}')
    # plt.savefig(os.path.join(depth_out_dir, f'frame_{i:03d}_depth.png'))
    mirror_mask = img[..., -1]
    mirror_mask = mirror_mask.numpy()
    mirror_out_dir = os.path.join(out_dir, 'mirror_mask')
    os.makedirs(mirror_out_dir, exist_ok=True)
    np.save(os.path.join(mirror_out_dir, f'frame_{i:03d}_mirror_mask.npy'), mirror_mask)
    # plt.figure()


# scene = mi.load_file(xml_path)
# print("Scene loaded.")


# parameters = mi.traverse(scene)
# parameters['camera.to_world'] = mi.ScalarTransform4f().look_at(
#     origin=[-3.0, 2.0, -0.5],
#     target=[-4.0, 2.0, -1.5],
#     up=[0.0, 1.0, 0.0]
# )
# parameters.update()

# import time
# start_time = time.time()
# image = mi.render(scene, spp=8)
# mi.util.write_bitmap('/Users/luigi/Projects/Research/Reconstruction Based on Removal/Data/Render/dining_room.png', image[...,:3])
# depth = image[...,3:4]
# depth = depth.numpy()
# print("Depth range:", depth.min(), depth.max())

# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(depth, cmap='plasma')
# plt.colorbar()
# plt.title('Depth Map')
# plt.show()


# mi.util.write_bitmap('/Users/luigi/Projects/Research/Reconstruction Based on Removal/Data/Render/dining_room_depth.png', image[...,3:4])
# mi.util.write_bitmap('/Users/luigi/Projects/Research/Reconstruction Based on Removal/Data/Render/dining_room_normal.png', (image[...,4:7] + 1.0) * 0.5)
# end_time = time.time()
# print(f"Render time: {end_time - start_time} seconds")
# print(f"Estimated render time for 90 frames at 512 spp: {(end_time - start_time) * 90 / 60:.2f} minutes")
# print(f"Estimated render time for 90 frames at 1024 spp: {(end_time - start_time) * 2 * 90 / 60:.2f} minutes")

# Create video from frames using ffmpeg
# (ffmpeg
#  .input(os.path.join(out_dir, 'frame_%03d.png'), framerate=30)
#  .output('/home/elec594/Desktop/luigi/Render/kitchen_animation.mp4')
#  .run())