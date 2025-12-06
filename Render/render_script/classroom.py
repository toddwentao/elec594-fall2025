import mitsuba as mi
import numpy as np
import os
import tqdm
import ffmpeg
import matplotlib.pyplot as plt


mi.set_variant('cuda_ad_rgb')

scene_path = '/home/elec594/Desktop/luigi/Render/mitsuba_gallery/classroom/scene.xml'
xml_path = scene_path

# X-axis: left window
# Y-axis: height
# Z-axis: back into the room

out_dir = '/home/elec594/Desktop/luigi/Render/classroom/flat_lens_frames'
os.makedirs(out_dir, exist_ok=True)

num_frames = 90
x = -1.0
y = 1.5
step_z = 0.03
radius = 0.5



for i in tqdm.tqdm(range(0, num_frames, 5), desc="Rendering frames"):
    z = -1.5 + step_z * i

    rotation_angle = (i / num_frames) * 0.5 * np.pi + 0.25 * np.pi  # Rotate 90 degrees over the sequence
    origin = [x, y, z]
    target = [x + np.sin(rotation_angle) * radius, y , z + np.cos(rotation_angle) * radius]

    scene = mi.load_file(xml_path)
    parameters = mi.traverse(scene)
    parameters['camera.to_world'] = mi.ScalarTransform4f().look_at(
        origin=origin,
        target=target,
        up=[0.0, 1.0, 0.0]
    )
    parameters.update()

    img = mi.render(scene, spp=64)  # reduce spp for faster preview
    # depth = img[..., 3]
    # depth = depth.numpy()
    # depth_out_dir = os.path.join(out_dir, 'depth')
    # os.makedirs(depth_out_dir, exist_ok=True)
    # np.save(os.path.join(depth_out_dir, f'frame_{i:03d}_depth.npy'), depth)
    # plt.imshow(depth, cmap='plasma')
    # plt.colorbar()
    # plt.title(f"Depth Map - Frame {i}")
    # plt.axis("off")
    # plt.savefig(os.path.join(depth_out_dir, f'frame_{i:03d}_depth.png'))
    # plt.close()

    mirror_mask = img[..., -1]
    mirror_mask = mirror_mask.numpy()
    mirror_out_dir = os.path.join(out_dir, 'mirror_mask')
    os.makedirs(mirror_out_dir, exist_ok=True)
    np.save(os.path.join(mirror_out_dir, f'frame_{i:03d}_mirror_mask.npy'), mirror_mask)
    plt.imshow(mirror_mask, cmap='gray')


exit()

scene = mi.load_file(xml_path)
print("Scene loaded.")


parameters = mi.traverse(scene)
parameters['camera.to_world'] = mi.ScalarTransform4f().look_at(
    origin=[-1.0, 1.5, 0.2],
    target=[1.0, 1.5, 0.1],
    up=[0.0, 1.0, 0.0]
)
parameters.update()

import time
start_time = time.time()
image = mi.render(scene, spp=8)
mi.util.write_bitmap('/home/elec594/Desktop/luigi/Render/classroom_output.png', image)
end_time = time.time()
print(f"Render time: {end_time - start_time} seconds")
print(f"Estimated render time for 90 frames at 512 spp: {(end_time - start_time) * 90 / 60:.2f} minutes")
print(f"Estimated render time for 90 frames at 1024 spp: {(end_time - start_time) * 2 * 90 / 60:.2f} minutes")

# Create video from frames using ffmpeg
# (ffmpeg
#  .input(os.path.join(out_dir, 'frame_%03d.png'), framerate=30)
#  .output('/home/elec594/Desktop/luigi/Render/classroom_animation.mp4')
#  .run())