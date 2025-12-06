import os, json, math
import numpy as np
import mitsuba as mi

mi.set_variant("scalar_rgb")

# ========= 可调参数 =========
OUT_DIR = "./Optical_Flow/mirror_scene"
IMG_W, IMG_H = 640, 480
FOV_DEG = 45.0          # 透视相机水平视场角（deg），可改
Z0 = 5.0                # 相机距离 z 轴深度（m）
ANGLES_DEG = [-5.0, 0.0, 5.0]   # 三帧角度（绕 y 轴），按需改

# 镜面/物体配置（与你现有场景一致）
def make_scene_and_pose(cam_angle_deg: float):
    # 根据角度，计算相机水平偏移（绕 y 轴）
    x_offset = Z0 * math.tan(math.radians(cam_angle_deg))
    origin = [x_offset, 0.0, Z0]
    target = [0.0, 0.0, 0.0]
    up     = [0.0, 1.0, 0.0]

    # 相机的世界←相机 4×4 变换（to_world）
    T_wc = mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up)

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path"},

        # 镜面：z=0 的矩形，法向 +Z
        "mirror": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate([0, 0, 0]) @ mi.ScalarTransform4f.scale([2, 2, 1]),
            "bsdf": {"type": "conductor", "material": "Ag"},
        },

        # 红色近物体
        "near_block": {
            "type": "cube",
            "to_world": (
                mi.ScalarTransform4f.rotate([0, 1, 0], 190)
                @ mi.ScalarTransform4f.scale([0.5, 2, 1])
                @ mi.ScalarTransform4f.translate([-0.3, 0, -8])
            ),
            "bsdf": {
                "type": "twosided",
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.8, 0.1, 0.1]}},
            },
        },

        # 蓝色远物体
        "far_block": {
            "type": "cube",
            "to_world": (
                mi.ScalarTransform4f.rotate([0, 1, 0], 175)
                @ mi.ScalarTransform4f.scale([0.5, 2, 1])
                @ mi.ScalarTransform4f.translate([0.3, 0, -9])
            ),
            "bsdf": {
                "type": "twosided",
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.1, 0.1, 0.8]}},
            },
        },

        # 柔和环境光
        "emitter": {"type": "constant", "radiance": {"type": "rgb", "value": [1.2, 1.2, 1.2]}},

        # 相机
        "sensor": {
            "type": "perspective",
            "to_world": T_wc,
            "fov": FOV_DEG,  # 指定水平 FOV（mitsuba 以横向为默认）
            "film": {"type": "hdrfilm", "width": IMG_W, "height": IMG_H, "rfilter": {"type": "gaussian"}},
            "sampler": {"type": "independent", "sample_count": 256},
        },
    }
    return scene_dict, T_wc, origin, target, up


def transform4f_to_numpy(T) -> np.ndarray:
    # Mitsuba 的 Transform4f.matrix 是列主序矩阵；转成 np.array 即可
    return np.array(T.matrix, dtype=np.float64)


def invert_4x4(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(M)


def decompose_extrinsics(T_wc_mat: np.ndarray):
    """
    从 T_world_from_cam（世界←相机）分解出：
      - R_wc, t_wc（相机在世界坐标中的姿态）
      - R_cw, t_cw（投影用：x ~ K [R_cw | t_cw] X_world）
    """
    R_wc = T_wc_mat[:3, :3]
    t_wc = T_wc_mat[:3, 3]

    # 相机←世界：T_cw = inv(T_wc)
    T_cw = np.linalg.inv(T_wc_mat)
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    return R_wc, t_wc, R_cw, t_cw, T_cw


def compute_intrinsics(fov_deg: float, w: int, h: int):
    # 假设像素方形 & fov 为水平视场角
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,   0,  1]], dtype=np.float64)
    return K, fx, fy, cx, cy


def ensure_dirs():
    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)


def main():
    ensure_dirs()

    # 1) 计算并保存内参
    K, fx, fy, cx, cy = compute_intrinsics(FOV_DEG, IMG_W, IMG_H)
    intrinsics = {
        "width": IMG_W, "height": IMG_H, "fov_deg": FOV_DEG,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "K": K.tolist(),
        "notes": "K assumes square pixels and horizontal FOV."
    }
    with open(os.path.join(OUT_DIR, "intrinsics.json"), "w") as f:
        json.dump(intrinsics, f, indent=2)

    # 2) 渲染每一帧并记录位姿
    frames = []
    for idx, ang in enumerate(ANGLES_DEG):
        scene_dict, T_wc, origin, target, up = make_scene_and_pose(ang)
        scene = mi.load_dict(scene_dict)
        img = mi.render(scene)

        # 保存 PNG
        frame_id = f"frame_{idx:03d}"
        png_path = os.path.join(OUT_DIR, "images", f"{frame_id}.png")
        bmp = mi.Bitmap(img).convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True,
        )
        bmp.write(png_path)

        # 变换矩阵
        sensor = scene.sensors()[0]
        try:
            T_wc_raw = sensor.world_transform()
        except TypeError:
            # Older Mitsuba builds expect an explicit time sample
            T_wc_raw = sensor.world_transform(0.0)
        T_wc_mat = transform4f_to_numpy(T_wc_raw)
        R_wc, t_wc, R_cw, t_cw, T_cw = decompose_extrinsics(T_wc_mat)

        frames.append({
            "id": frame_id,
            "angle_deg": float(ang),
            "image_path": png_path,
            "origin": origin,
            "target": target,
            "up": up,
            "T_world_from_cam": T_wc_mat.tolist(),
            "T_cam_from_world": T_cw.tolist(),
            "R_wc": R_wc.tolist(),
            "t_wc": t_wc.tolist(),
            "R_cw": R_cw.tolist(),
            "t_cw": t_cw.tolist(),
        })

    # 3) 写 poses.json（含约定说明）
    poses = {
        "conventions": {
            "projection": "x ~ K [R_cw | t_cw] X_world",
            "T_world_from_cam": "4x4 transform mapping camera coordinates to world coordinates",
            "T_cam_from_world": "inverse of T_world_from_cam",
            "note": "R_cw,t_cw 用于投影；R_wc,t_wc 表示相机在世界系中的位姿。"
        },
        "frames": frames
    }
    with open(os.path.join(OUT_DIR, "poses.json"), "w") as f:
        json.dump(poses, f, indent=2)

    print(f"[OK] Wrote {len(frames)} images, intrinsics.json and poses.json to: {OUT_DIR}")

if __name__ == "__main__":
    main()
