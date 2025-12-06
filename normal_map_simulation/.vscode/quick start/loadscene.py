import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant("llvm_ad_rgb")
scene = mi.load_file("scenes/simple.xml")

original_image = mi.render(scene, spp=128)
plt.axis("off")
plt.imshow(original_image ** (1.0 / 2.2))
mi.util.write_bitmap("outputs/original_image.png",original_image)

params = mi.traverse(scene)
print(params)

print('sensor.near_clip:             ',  params['sensor.near_clip'])
print('teapot.bsdf.reflectance.value:',  params['teapot.bsdf.reflectance.value'])
print('light1.intensity.value:       ',  params['light1.intensity.value'])
print('light2.intensity.value:       ',  params['light2.intensity.value'])

# Give a red tint to light1 and a green tint to light2
params['light1.intensity.value'] *= [1.5, 0.2, 0.2]
params['light2.intensity.value'] *= [0.2, 1.5, 0.2]

# Apply updates
params.update()

print('light1.intensity.value:       ',  params['light1.intensity.value'])
print('light2.intensity.value:       ',  params['light2.intensity.value'])

# Translate the teapot a little bit
V = dr.unravel(mi.Point3f, params['teapot.vertex_positions'])
V.z += 0.5
params['teapot.vertex_positions'] = dr.ravel(V)

# Apply changes
params.update()

modified_image = mi.render(scene, spp=128)

fig = plt.figure(figsize=(10, 10))
fig.add_subplot(1,2,1).imshow(original_image); plt.axis('off'); plt.title('original')
fig.add_subplot(1,2,2).imshow(modified_image); plt.axis('off'); plt.title('modified')

fig.savefig("outputs/image_mod_ori.png")

print("已运行完")