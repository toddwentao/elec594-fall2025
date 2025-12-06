import mitsuba as mi
import numpy as np
import os
import tqdm
import ffmpeg


mi.set_variant('cuda_ad_rgb')

scene_path = '/home/elec594/Desktop/luigi/Render/mitsuba_gallery/kitchen/scene.xml'
xml_path = scene_path

# X-axis: left 
# Y-axis: up
# Z-axis: backward

out_dir = '/home/elec594/Desktop/luigi/Render/kitchen/frames'
os.makedirs(out_dir, exist_ok=True)

num_frames = 170
init_x = -0.8
init_y = 1.5
init_z = 1.2
step_x = 0.01
step_y = 0.0
step_z = -0.01
radius = 0.01

for i in tqdm.tqdm(range(0, num_frames, 1), desc="Rendering frames"):
    x = init_x + step_x * i
    y = init_y + step_y * i
    z = init_z + step_z * i

    rotation_angle = - (i / num_frames) * 0.4 * np.pi + 1.4 * np.pi  # Slight rotation over the sequence
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
    mi.util.write_bitmap(os.path.join(out_dir, f'frame_{i:03d}.png'), img)

scene = mi.load_file(xml_path)
print("Scene loaded.")


# parameters = mi.traverse(scene)
# parameters['camera.to_world'] = mi.ScalarTransform4f().look_at(
#     origin=[0.8, 1.5, -0.5],
#     target=[0.3, 1.5, -1.5],
#     up=[0.0, 1.0, 0.0]
# )
# parameters.update()

import time
start_time = time.time()
image = mi.render(scene, spp=8)
mi.util.write_bitmap('/home/elec594/Desktop/luigi/Render/kitchen_output.png', image)
end_time = time.time()
print(f"Render time: {end_time - start_time} seconds")
# print(f"Estimated render time for 90 frames at 512 spp: {(end_time - start_time) * 90 / 60:.2f} minutes")
# print(f"Estimated render time for 90 frames at 1024 spp: {(end_time - start_time) * 2 * 90 / 60:.2f} minutes")

# Create video from frames using ffmpeg
# (ffmpeg
#  .input(os.path.join(out_dir, 'frame_%03d.png'), framerate=30)
#  .output('/home/elec594/Desktop/luigi/Render/kitchen_animation.mp4')
#  .run())