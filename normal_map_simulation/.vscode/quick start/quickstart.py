import mitsuba as mi
import matplotlib.pyplot as plt

mi.variants()
mi.set_variant("scalar_rgb")
scene = mi.load_file("scenes/cbox.xml")
image = mi.render(scene,spp=256)

plt.axis("off")
plt.imshow(image ** (1.0 / 2.2))

mi.util.write_bitmap("outputs/my_first_render.png",image)
mi.util.write_bitmap("outputs/my_first_render.exr", image)