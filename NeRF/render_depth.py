import torch
import numpy as np
import imageio
from pathlib import Path

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.io import load_from_json


# ================================================================
# 0. 用户路径设置（你只需要改这三条）
# ================================================================
CONFIG_PATH = Path("/Users/young/Documents/matata/code/NeRF/Output_Colab_flat_90/config.yml")
TRANSFORMS_PATH = Path("/Users/young/Documents/matata/code/NeRF/datasets/flat_lens_frames/transforms.json")
DATAPARSER_PATH = Path("/Users/young/Documents/matata/code/NeRF/Output_Colab_flat_90/dataparser_transforms.json")

OUTPUT_PATH = "depth_fixed.png"


# ================================================================
# 1. Mitsuba → Nerfstudio 坐标系转换（关键修复）
# ================================================================
def mitsuba_to_nerfstudio(c2w):
    """
    Mitsuba uses +Z forward
    Nerfstudio uses -Z forward
    """
    fix = torch.eye(4)
    fix[2, 2] = -1   # flip Z axis (forward axis)
    return c2w @ fix


# ================================================================
# 2. Load Nerfstudio model
# ================================================================
config, pipeline, _, _ = eval_setup(CONFIG_PATH, test_mode="inference")


# ================================================================
# 3. Load Mitsuba transforms.json   (真实相机位姿)
# ================================================================
meta = load_from_json(TRANSFORMS_PATH)
frame = meta["frames"][0]        # 你可以改成 1、2、3 取其他视角
h, w = meta["h"], meta["w"]

c2w_raw = torch.tensor(frame["transform_matrix"], dtype=torch.float32)


# ================================================================
# 4. Load dataparser_transforms.json   (Nerfstudio normalize)
# ================================================================
dp = load_from_json(DATAPARSER_PATH)

T = torch.tensor(dp["transform"], dtype=torch.float32)    # 3×4
scale = dp["scale"]

# 3×4 → 4×4
T4 = torch.eye(4)
T4[:3, :4] = T


# ================================================================
# 5. 正确变换相机位姿：Mitsuba -> NS canonical
# ================================================================
# Step 1: 修复 Mitsuba 相机朝向 (+Z → -Z)
c2w_fixed = mitsuba_to_nerfstudio(c2w_raw)

# Step 2: 应用 Nerfstudio normalization
c2w_ns = (c2w_fixed @ T4) / scale

# 取前三行作为 3×4 矩阵
c2w_final = c2w_ns[:3, :4].unsqueeze(0)   # shape = (1, 3, 4)


# ================================================================
# 6. 构建 Nerfstudio 相机对象
# ================================================================
camera = Cameras(
    fx = meta["fl_x"],
    fy = meta["fl_y"],
    cx = meta["cx"],
    cy = meta["cy"],
    width  = w,
    height = h,
    camera_to_worlds = c2w_final,
)


# ================================================================
# 7. Render depth
# ================================================================
with torch.no_grad():
    outputs = pipeline.model.get_outputs_for_camera(camera)

depth = outputs["depth"][0].cpu().numpy()


# ================================================================
# 8. Normalize & Save
# ================================================================
depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
depth_img = (depth_norm * 255).astype(np.uint8)

imageio.imwrite(OUTPUT_PATH, depth_img)
print(f"Depth saved to {OUTPUT_PATH}")
print(f"Depth shape = {depth.shape}")