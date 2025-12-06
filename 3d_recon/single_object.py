import mitsuba as mi
import os, math
import numpy as np

mi.set_variant('scalar_rgb')

# basic parameters
# ----------------------
# BACKGROUND_PNG = 'scenes/textures/flower_photo.jpeg'  # 背景图片（PNG/JPEG）
BACKGROUND_PNG = 'scenes/textures/pattern_10_lines.png'  # 背景图片（PNG/JPEG）
OUTPUT_DIR = 'output/single_object/pattern_10_lines'
# OUTPUT_DIR = 'output/single_object_renders'
NUM_FRAMES = 120 # 帧数
FPS = 30 # time =num_frames / fps

CAM_SWEEP = 3   # X 方向摆动幅度（可按画面调大/调小）
CAMERA_HEIGHT = 0.0  # 摄像机高度（z）
CAMERA_FOV = 40.0

# background
BACKGROUND_SCALE = 3  # 背景矩形的缩放（越大越稳）

# 按你的约定：Z 向上，X 朝屏幕外，Y 向右
OBJECT_POS = [0.0, 0.0, 0.0]
GLASS_SCALE = [2, 2, 2]   # 矩形宽/高（z 轴对平面无厚度影响）

# 按你的要求：从 plane 沿 +Y 放置 camera 和背景
CAMERA_OFFSET_FROM_PLANE = 2  # 摄像机在 plane 
BACKGROUND_OFFSET_FROM_CAMERA = 1 # 背景在 camera 再 +1.0m


# Object
OBJ_FILE = 'scenes/my_meshes/distorted_plane.obj'


# 渲染
SAMPLES = 128
W, H = 256, 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

def camera_sweep_x(frame_idx: int):
    if NUM_FRAMES <= 1:
        t = 0.0
    else:
        t = frame_idx / (NUM_FRAMES - 1)  # 0 ~ 1
    x = OBJECT_POS[0] + (-CAM_SWEEP + 2*CAM_SWEEP*t)
    y = OBJECT_POS[1] + CAMERA_OFFSET_FROM_PLANE
    z = OBJECT_POS[2] + CAMERA_HEIGHT

    origin = [x, y, z]
    target = [x,y-1.0,z]
    return mi.ScalarTransform4f.look_at(origin=origin, target=target, up=[0,0,1])

def extract_intrinsics_extrinsics(sensor):
    """
    Extracts intrinsics K and extrinsics (R, t) from a Mitsuba 3 sensor.
    """

    # ----------------------------------------------------------
    # 1. EXTRINSICS
    # ----------------------------------------------------------
    T_c2w = sensor.world_transform().matrix         # 4×4 camera-to-world
    T_w2c = np.linalg.inv(T_c2w)                    # world-to-camera (extrinsic)

    R = T_w2c[:3, :3]
    t = T_w2c[:3, 3]

    # Camera origin in world = T_c2w * [0,0,0,1]
    origin_h = T_c2w @ np.array([0, 0, 0, 1])
    origin = origin_h[:3]

    # ----------------------------------------------------------
    # 2. INTRINSICS
    # ----------------------------------------------------------
    film = sensor.film()
    W, H = film.crop_size()

    # Pixel centers
    px = W * 0.5
    py = H * 0.5

    # Norm. pixel coords
    u_center = [px / W, py / H]
    u_right  = [(px + 1) / W, py / H]
    u_up     = [px / W, (py + 1) / H]

    u_center = mi.Point2f(px / W, py / H)
    u_right  = mi.Point2f((px+1)/W, py / H)
    u_up     = mi.Point2f(px/W, (py+1)/H)

    # Sample rays (time=0.0, sample1=random float 0-1, sample2=random 2D point, sample3 random 2D point)
    # For our purpose, sample1=0, sample2=u, sample3=(0,0)
    center_ray, _ = sensor.sample_ray(time=0.0, sample1=0.0, sample2=u_center, sample3=mi.Point2f(0,0))
    right_ray, _  = sensor.sample_ray(time=0.0, sample1=0.0, sample2=u_right,  sample3=mi.Point2f(0,0))
    up_ray, _     = sensor.sample_ray(time=0.0, sample1=0.0, sample2=u_up,     sample3=mi.Point2f(0,0))

    # Directions
    d0 = np.array(center_ray.d)
    dx = np.array(right_ray.d)
    dy = np.array(up_ray.d)

    # Compute fx, fy from angle between rays
    theta_x = np.arccos(np.clip(np.dot(d0, dx), -1, 1))
    theta_y = np.arccos(np.clip(np.dot(d0, dy), -1, 1))

    fx = 1.0 / np.tan(theta_x)
    fy = 1.0 / np.tan(theta_y)

    cx = W / 2
    cy = H / 2

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,   1]
    ])

    return K, R, t, T_c2w, T_w2c, origin

def create_scene_dict(frame_idx: int):

    """
    坐标系:Z 向上,X 出屏（相机朝向) Y 向右
    目标布局（沿 +Y 由近到远):object  ->  camera  ->  background
    相机：来自 +Y 看向 -Y(up = Z)
    背景：放在更远的 +Y,法线 = -Y,作为面光源使用 BACKGROUND_PNG
    """

    # --- 2) 背景板 world 变换：默认 rectangle 法线是 +Z
        # 让它面向相机（来自 +Y 的相机），需要把法线旋到 -Y：绕 X 轴旋 -90°
    bg_to_world = (
        mi.ScalarTransform4f.translate([
            OBJECT_POS[0],
            OBJECT_POS[1] + CAMERA_OFFSET_FROM_PLANE + BACKGROUND_OFFSET_FROM_CAMERA,  # +Y 更远
            OBJECT_POS[2]
        ])
        @ mi.ScalarTransform4f.rotate([1, 0, 0], +90.0) # +Z → +Y（正面朝 +Y）
        @ mi.ScalarTransform4f.scale([BACKGROUND_SCALE, BACKGROUND_SCALE, 1.0])

        )

    # --- 3) 玻璃几何 world 变换（注意别零尺度）---
    glass_to_world = (
        
        mi.ScalarTransform4f.translate([OBJECT_POS[0], 0.0, OBJECT_POS[2]])
        # @ mi.ScalarTransform4f.rotate([1,0,0], -90.0) # 让法线 = +Y（面向相机）
        @ mi.ScalarTransform4f.scale(GLASS_SCALE)
       
    )

    

    # --- 4) 组场景 ---
    scene = {
        'type': 'scene',
        'integrator': {'type': 'path'},

        'emitter': {
                        'type': 'constant',
                        'radiance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}  # 均匀白光
                    },

        # 3D 物体 镜面
        'glass': {
            'type': 'obj',
            # 'type':'rectangle',
            'filename': OBJ_FILE,
            'to_world':glass_to_world,
            'bsdf': {'type': 'conductor',
                     'material': 'Al',
                    #  'alpha': 0.02,
                    #  'distribution': 'ggx',
                    },
            'face_normals': True,
            'flip_normals': True,  # 如果渲出来还是背面，改成 True 再试
                },
            # 'bsdf': { 'type': 'diffuse', 'reflectance': {'type':'rgb','value':[0.1,0.9,0.1]} }
                

        # 背景：使用一个矩形平面放在 camera 的前方（沿 +Y），法线朝 -Y
        'background': {
            'type': 'rectangle',
            'to_world': bg_to_world,
            'bsdf': {
                'type':'twosided',
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'bitmap',
                        'filename': BACKGROUND_PNG, # png贴图到背景矩形里
                        'to_uv': mi.ScalarTransform3f.rotate(180),
                    }}
                },

            # 'emitter': {
            #     'type': 'area',
            #     'radiance': {
            #         'type': 'bitmap',
            #         'filename': BACKGROUND_PNG,
            #         # 若方向不对，再开这一行做UV旋转（试 90/-90/180）
            #         # 'transform': mi.ScalarTransform3f.rotate([0,0,1], 90)
            #     }
            # },
        },

        # 相机（按帧放置）
        'sensor': {
            'type': 'perspective',
            'fov': CAMERA_FOV,
            'to_world': camera_sweep_x(frame_idx),
            # 'to_world': mi.ScalarTransform4f.look_at(
            #     origin=[OBJECT_POS[0], OBJECT_POS[1] + CAMERA_OFFSET_FROM_PLANE, OBJECT_POS[2] + CAMERA_HEIGHT],
            #     target=[OBJECT_POS[0], OBJECT_POS[1] + CAMERA_OFFSET_FROM_PLANE -1.0, OBJECT_POS[2] + CAMERA_HEIGHT],  # 面向 −Y
            #     up=[0,0,1]
            # ),  
            'sampler': {'type': 'independent', 'sample_count': SAMPLES},
            # 'film': {
            #         'type': 'hdrfilm',
            #         'width': W,
            #         'height': H,
            #         'pixel_format': 'rgb',           # 可留着
            #         'rfilter': { 'type': 'box' },    # 预览更快
            #         'sample_border': False,          # 可选，避免边缘加采样
            #         'banner': False                  # 可选，去掉角落的小字
            #     }
            'film': {'type': 'hdrfilm', 'width': W, 'height': H}

        }
    }

    cam_y = OBJECT_POS[1] + CAMERA_OFFSET_FROM_PLANE
    bg_y = OBJECT_POS[1] + CAMERA_OFFSET_FROM_PLANE + BACKGROUND_OFFSET_FROM_CAMERA
    print(f"[dbg] y-order: bg={bg_y:.3f} > cam={cam_y:.3f} > mirror=0.000")
    print("[dbg] bg_to_world =\n", bg_to_world)
    print("[dbg] glass_to_world =\n", glass_to_world)

    return scene


def main():
    assert os.path.exists(BACKGROUND_PNG), f"找不到贴图: {BACKGROUND_PNG}"
    print('开始渲染 single_object 场景...')
    for i in range(NUM_FRAMES):
        scene_dict = create_scene_dict(i)
        
        scene = mi.load_dict(scene_dict)
        sensor = scene.sensors()[0]

        K, R, t, T_c2w, T_w2c, origin = extract_intrinsics_extrinsics(sensor)
        with open(os.path.join(OUTPUT_DIR, f'frame_{i:04d}_camera_params.txt'), 'w') as f:
            f.write("=== CAMERA PARAMETERS ===\n\n")
            f.write("K (intrinsics):\n")
            f.write(str(K) + "\n\n")
            f.write("R (rotation world→camera):\n")
            f.write(str(R) + "\n\n")
            f.write("t (translation world→camera):\n")
            f.write(str(t) + "\n\n")
            f.write("Camera origin (world coords):\n")
            f.write(str(origin) + "\n\n")
            f.write("T_c2w (camera→world):\n")
            f.write(str(T_c2w) + "\n\n")
            f.write("T_w2c (world→camera):\n")
            f.write(str(T_w2c) + "\n\n")
        # print("\n=== CAMERA PARAMETERS ===")
        # print("K (intrinsics):\n", K)
        # print("R (rotation world→camera):\n", R)
        # print("t (translation world→camera):\n", t)
        # print("Camera origin (world coords):", origin)
        # print("T_c2w (camera→world):\n", T_c2w)
        # print("T_w2c (world→camera):\n", T_w2c)
        
        img = mi.render(scene, seed=42)

        bmp = mi.Bitmap(img)
        bmp8 = bmp.convert(pixel_format=mi.Bitmap.PixelFormat.RGB,
                           component_format=mi.Struct.Type.UInt8,
                           srgb_gamma=True)

        out_path = os.path.join(OUTPUT_DIR, f'frame_{i:04d}.png')
        bmp8.write(out_path)
        print(f'  已保存: {out_path}')

    print('渲染完成。')

    # --- 4. 合并帧序列为视频 ---
    print("使用 FFmpeg 合并视频中...")
    ffmpeg_command = f'ffmpeg -y -r {FPS} -i {OUTPUT_DIR}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" {OUTPUT_DIR}/final_reflection_animation.mp4'
    os.system(ffmpeg_command)
    print("视频合并完成！文件名为 final_reflection_animation.mp4")

    


if __name__ == '__main__':
    main()
