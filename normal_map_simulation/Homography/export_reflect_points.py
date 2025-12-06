from Homography.create_multiobject_demo import make_scene
import mitsuba as mi
import numpy as np

mi.set_variant("scalar_rgb")
scene_dict = make_scene(cam_angle_deg=10.0)
scene = mi.load_dict(scene_dict)
sensor = scene.sensors()[0]
film = sensor.film()
width, height = film.size()

reflect_pts = np.zeros((height, width, 3))
mirror_pts = np.zeros((height, width, 3))
normals = np.zeros((height, width, 3))

sample_pos = mi.Point2f(0.5, 0.5)

for y in range(height):
    for x in range(width):
        film_uv = mi.Point2f((x + 0.5) / width, (y + 0.5) / height)
        ray, _ = sensor.sample_ray(0.0, 0.0, film_uv, sample_pos)
        si = scene.ray_intersect(ray)
        if not si.is_valid():
            continue
        mirror_pts[y, x, :] = si.p
        normals[y, x, :] = si.n
        d_reflect = mi.reflect(ray.d, si.n)
        refl_ray = mi.Ray3f(si.p + 1e-4 * si.n, d_reflect)
        si2 = scene.ray_intersect(refl_ray)
        if si2.is_valid():
            reflect_pts[y, x, :] = si2.p

np.save("./Homography/mirror_scene/mirror_pts.npy", mirror_pts)
np.save("./Homography/mirror_scene/reflect_pts.npy", reflect_pts)
np.save("./Homography/mirror_scene/normals.npy", normals)
print("✅ 导出完成！")
