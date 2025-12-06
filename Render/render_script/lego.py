import mitsuba as mi
import numpy as np
import os
import tqdm

mi.set_variant('cuda_ad_rgb')

scene_path = '/home/elec594/Desktop/luigi/Render/lego/scene.xml'
xml_path = scene_path


out_dir = '/home/elec594/Desktop/luigi/Render/lego/frames'
os.makedirs(out_dir, exist_ok=True)

num_frames = 60
x = 0.1
step_y = 0.005
step_z = 0.005

for i in tqdm.tqdm(range(num_frames), desc="Rendering frames"):
    y = 0.1 + step_y * i
    z = 0.1 + step_z * i

    scene = mi.load_file(xml_path)
    parameters = mi.traverse(scene)
    parameters['camera.to_world'] = mi.ScalarTransform4f().look_at(
        origin=[x, y, z],
        target=[-0.2, 0.3, 0.3],
        up=[0.0, 1.0, 0.0]
    )
    parameters.update()

    img = mi.render(scene, spp=256)  # reduce spp for faster preview
    mi.util.write_bitmap(os.path.join(out_dir, f'frame_{i:03d}.png'), img)

scene = mi.load_file(xml_path)
print("Scene loaded.")

image = mi.render(scene, spp=16)
mi.util.write_bitmap('/home/elec594/Desktop/luigi/Render/lego_output.png', image)