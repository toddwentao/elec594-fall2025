import mitsuba as mi
import os
import numpy as np

mi.set_variant('scalar_rgb')

# basic parameters
# ----------------------
BACKGROUND_COLOR = [0.02, 0.02, 0.02]  # 暗背景
OUTPUT_DIR = 'output/lamp_renders'
NUM_FRAMES = 70
FPS = 30

CAM_SWEEP = 3
CAMERA_HEIGHT = 0.0
CAMERA_FOV = 40.0

BACKGROUND_SCALE = 12

OBJECT_POS = [0.0, 0.0, 0.0]
GLASS_SCALE = [2, 2, 2]

CAMERA_OFFSET_FROM_PLANE = 2
BACKGROUND_OFFSET_FROM_CAMERA = 2.1  # 让背景在相机和路灯之后更远处

LAMP_POSITION = [0.0, CAMERA_OFFSET_FROM_PLANE + 2.0, 0.0]  # 放在相机之后，通过镜面反射可见
LAMP_SCALE = [0.08, 0.08, 4.4]
LAMP_COLOR = [1.0, 1.0, 1.0] # 白色
LAMP_EMISSION = [10.0, 10.0, 10.0]

OBJ_FILE = 'scenes/my_meshes/distorted_plane.obj'

SAMPLES = 128
W, H = 256, 256

os.makedirs(OUTPUT_DIR, exist_ok=True)


def camera_sweep_x(frame_idx: int):
    if NUM_FRAMES <= 1:
        t = 0.0
    else:
        t = frame_idx / (NUM_FRAMES - 1)
    x = OBJECT_POS[0] + (-CAM_SWEEP + 2 * CAM_SWEEP * t)
    y = OBJECT_POS[1] + CAMERA_OFFSET_FROM_PLANE
    z = OBJECT_POS[2] + CAMERA_HEIGHT

    origin = [x, y, z]
    target = [x, y - 1.0, z]
    return mi.ScalarTransform4f.look_at(origin=origin, target=target, up=[0, 0, 1])


def create_scene_dict(frame_idx: int):
    """
    坐标系:Z 向上,X 出屏，Y 向右
    布局（沿 +Y 由近到远):object -> camera -> background
    """
    if NUM_FRAMES <= 1:
        t = 0.0
    else:
        t = frame_idx / (NUM_FRAMES - 1)
    cam_x = OBJECT_POS[0] + (-CAM_SWEEP + 2 * CAM_SWEEP * t)

    bg_to_world = (
        mi.ScalarTransform4f.translate([
            cam_x,
            OBJECT_POS[1] + CAMERA_OFFSET_FROM_PLANE + BACKGROUND_OFFSET_FROM_CAMERA,
            OBJECT_POS[2]
        ])
        @ mi.ScalarTransform4f.rotate([1, 0, 0], +90.0)
        @ mi.ScalarTransform4f.scale([BACKGROUND_SCALE, BACKGROUND_SCALE, 1.0])
    )

    glass_to_world = (
        mi.ScalarTransform4f.translate([OBJECT_POS[0], 0.0, OBJECT_POS[2]])
        @ mi.ScalarTransform4f.scale(GLASS_SCALE)
    )

    lamp_to_world = (
        mi.ScalarTransform4f.translate(LAMP_POSITION)
        @ mi.ScalarTransform4f.scale(LAMP_SCALE)
    )

    scene = {
        'type': 'scene',
        'integrator': {'type': 'path'},

        'emitter': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [0.0, 0.0, 0.0]}
        },

        'glass': {
            'type': 'obj',
            'filename': OBJ_FILE,
            'to_world': glass_to_world,
            'bsdf': {
                'type': 'conductor',
                'material': 'Al',
            },
            'face_normals': True,
            'flip_normals': True,
        },

        'lamp': {
            'type': 'cube',
            'to_world': lamp_to_world,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': LAMP_COLOR}
            },
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': LAMP_EMISSION}
            }
        },

        'background': {
            'type': 'rectangle',
            'to_world': bg_to_world,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': BACKGROUND_COLOR}
            },
        },

        'sensor': {
            'type': 'perspective',
            'fov': CAMERA_FOV,
            'to_world': camera_sweep_x(frame_idx),
            'sampler': {'type': 'independent', 'sample_count': SAMPLES},
            'film': {'type': 'hdrfilm', 'width': W, 'height': H}
        }
    }

    cam_y = OBJECT_POS[1] + CAMERA_OFFSET_FROM_PLANE
    bg_y = OBJECT_POS[1] + CAMERA_OFFSET_FROM_PLANE + BACKGROUND_OFFSET_FROM_CAMERA
    lamp_y = LAMP_POSITION[1]
    print(f"[dbg] cam_x={cam_x:.3f}")
    print(f"[dbg] y-order: cam={cam_y:.3f}, mirror=0.000, lamp={lamp_y:.3f}, bg={bg_y:.3f}")
    print("[dbg] bg_to_world =\n", bg_to_world)
    print("[dbg] glass_to_world =\n", glass_to_world)
    print("[dbg] lamp_to_world =\n", lamp_to_world)

    return scene


def main():
    print('开始渲染 single_object_lamp 场景...')
    for i in range(NUM_FRAMES):
        scene_dict = create_scene_dict(i)
        scene = mi.load_dict(scene_dict)

        img = mi.render(scene, seed=42)

        bmp = mi.Bitmap(img)
        bmp8 = bmp.convert(pixel_format=mi.Bitmap.PixelFormat.RGB,
                           component_format=mi.Struct.Type.UInt8,
                           srgb_gamma=True)

        out_path = os.path.join(OUTPUT_DIR, f'frame_{i:04d}.png')
        bmp8.write(out_path)
        print(f'  已保存: {out_path}')

    print('渲染完成。')

    print("使用 FFmpeg 合并视频中...")
    ffmpeg_command = f'ffmpeg -y -r {FPS} -i {OUTPUT_DIR}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" lamp_reflection_animation.mp4'
    os.system(ffmpeg_command)
    print("视频合并完成！文件名为 lamp_reflection_animation.mp4")


if __name__ == '__main__':
    main()
