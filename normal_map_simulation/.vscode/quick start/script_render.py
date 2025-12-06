import mitsuba as mi
import drjit as dr

mi.set_variant('llvm_ad_rgb')

scene = mi.load_file('../scenes/cbox.xml')
image = mi.render(scene,spp=256)

# Camera origin in world space
cam_origin = mi.Point3f(0, 1, 3)

# Camera view direction in world space
cam_dir = dr.normalize(mi.Vector3f(0, -0.5, -1))

# Camera width and height in world space
cam_width  = 2.0
cam_height = 2.0

# Image pixel resolution
image_res = (256, 256)


# Construct a grid of 2D coordinates
x, y = dr.meshgrid(
    dr.linspace(mi.Float, -cam_width  / 2,   cam_width / 2, image_res[0]),
    dr.linspace(mi.Float, -cam_height / 2,  cam_height / 2, image_res[1])
)

# Ray origin in local coordinates
ray_origin_local = mi.Vector3f(x, y, 0)

# Ray origin in world coordinates
ray_origin = mi.Frame3f(cam_dir).to_world(ray_origin_local) + cam_origin

ray = mi.Ray3f(o=ray_origin, d=cam_dir)

si = scene.ray_intersect(ray)