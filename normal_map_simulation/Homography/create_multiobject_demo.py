import mitsuba as mi
mi.set_variant("scalar_rgb")
import numpy as np
import os

output_dir = "./Homography/mirror_scene/"
os.makedirs(output_dir, exist_ok=True)

# 问题描述：
# 物体的z轴方向问题，为什么要翻向 -z

# -----------------------------
# Scene description
# -----------------------------
def make_scene(cam_angle_deg=0.0):
    # 固定相机Z位置
    z0 = 5.0

    # 根据角度自动计算相机的X偏移量
    x_offset = z0 * np.tan(np.deg2rad(cam_angle_deg))

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path"},

        # === 平面镜 ===


        "mirror": {
            "type": "rectangle",
            "to_world": (
                mi.ScalarTransform4f.translate([0, 0, 0])           # z=0 平面
                @ mi.ScalarTransform4f.scale([2, 2, 1])             # 默认法向 +Z → 朝相机
            ),
            "bsdf": {"type": "conductor", "material": "Ag"},
            # "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5,0.5,0.5]}}
            },

        # "bg": {
        #     "type": "rectangle",
        #     "to_world": mi.ScalarTransform4f.translate([0,0,0])
        #     @ mi.ScalarTransform4f.scale([2,2,1]),
        #     "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [1,0,0]}}
        #     },

        # === 红色矩形（近处物体） ===
        "near_block": {
            "type": "cube",
            "to_world": (
                mi.ScalarTransform4f.rotate([0, 1, 0], 190)
                @ mi.ScalarTransform4f.scale([0.5, 2, 1])
                @ mi.ScalarTransform4f.translate([-0.3, 0, -8])
            ),
            "bsdf": {
                "type": "twosided",
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": [0.8, 0.1, 0.1]}
                }
            },
        },

        # === 蓝色矩形（远处物体） ===
        "far_block": {
            "type": "cube",
            "to_world": (
                mi.ScalarTransform4f.rotate([0, 1, 0], 175)
                @ mi.ScalarTransform4f.scale([0.5, 2, 1])
                @ mi.ScalarTransform4f.translate([0.3, 0, -9])
            ),
        #     "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.1, 0.1, 0.8]}},
        # },
            "bsdf": {
                "type": "twosided",
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": [0.1, 0.1, 0.8]}
                }
            },
        },
        
        # === 主顶光（在蓝块上方，朝下照亮场景） ===
        # "main_light": {
        #     "type": "rectangle",
        #     "to_world": (
        #         mi.ScalarTransform4f.translate([0, 1.5, 2.6])  # 在蓝块上方
        #         @ mi.ScalarTransform4f.rotate([1, 0, 0], -90)  # 朝下照射
        #         @ mi.ScalarTransform4f.scale([6, 6, 1])
        #     ),
        #     "emitter": {
        #         "type": "area",
        #         "radiance": {"type": "rgb", "value": [60.0, 60.0, 60.0]},
        #     },
        # },

        # === 环境光（柔和填充） ===
        "emitter": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [1.2, 1.2, 1.2]},
        },

        # === 主前方光（镜子与物体之间，朝 +Z 照亮物体正面） ===
        # "front_key_light": {
        #     "type": "rectangle",
        #     "to_world": (
        #         mi.ScalarTransform4f.translate([0, 0.0, 2.6])   # 在镜子(0)与物体(3,4)之间
        #         @ mi.ScalarTransform4f.scale([6, 6, 1])         # 默认法线 +Z → 朝向物体
        #     ),
        #     "emitter": {
        #         "type": "area",
        #         "radiance": {"type": "rgb", "value": [60.0, 60.0, 60.0]},
        #     },
        # },

        # === 后方补光（在物体后面，朝 −Z 打过来） ===
        # "back_fill_light": {
        #     "type": "rectangle",
        #     "to_world": (
        #         mi.ScalarTransform4f.translate([0, 0.0, 5.5])
        #         @ mi.ScalarTransform4f.rotate([0, 1, 0], 180)
        #         @ mi.ScalarTransform4f.scale([8, 8, 1])
        #     ),
        #     "emitter": {
        #         "type": "area",
        #         "radiance": {"type": "rgb", "value": [20.0, 20.0, 20.0]},
        #     },
        # },

        # === 相机 ===
        "sensor": {
            "type": "perspective",   # ✅ 改成 perspective 相机
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[x_offset, 0, z0],
                target=[0, 0, 0],
                up=[0, 1, 0],
            ),
            "film": {
                "type": "hdrfilm",
                "width": 640,
                "height": 480,
                "rfilter": {"type": "gaussian"},
            },
            "sampler": {"type": "independent", "sample_count": 256},
        },
    }
    return scene_dict

# 渲染两帧相机视角，绕中心旋转±10度
for angle in [-10.0, 10.0]:
    scene = mi.load_dict(make_scene(cam_angle_deg=angle))
    img = mi.render(scene)
    base_name = os.path.join(output_dir, f"render_cam{angle:+.2f}")
    # exr_path = f"{base_name}.exr"
    png_path = f"{base_name}.png"

    bitmap = mi.Bitmap(img)
    # bitmap.write(exr_path)

    png_bitmap = bitmap.convert(
        pixel_format=mi.Bitmap.PixelFormat.RGB,
        component_format=mi.Struct.Type.UInt8,
        srgb_gamma=True,
    )
    png_bitmap.write(png_path)
