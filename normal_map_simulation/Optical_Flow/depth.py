import json, cv2, numpy as np, matplotlib.pyplot as plt

# === 加载相机参数 ===
with open('Optical_Flow/mirror_scene/intrinsics.json') as f:
    intr = json.load(f)
K = np.array(intr['K'])
fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']

with open('Optical_Flow/mirror_scene/poses.json') as f:
    poses = json.load(f)['frames']

R1 = np.array(poses[0]['R_cw'])
t1 = np.array(poses[0]['t_cw']).reshape(3,1)
R2 = np.array(poses[1]['R_cw'])
t2 = np.array(poses[1]['t_cw']).reshape(3,1)

R = R2 @ R1.T
t = t2 - R @ t1
tx = abs(t[0,0])  # 仅考虑水平平移

# === 读图 & 计算光流 ===
img1 = cv2.imread('Optical_Flow/mirror_scene/images/frame_000.png')
img2 = cv2.imread('Optical_Flow/mirror_scene/images/frame_001.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                    pyr_scale=0.5, levels=3, winsize=15,
                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
u = flow[...,0]

# === 计算相对深度 ===
mask = np.abs(u) > 1.0  # 过滤静态背景
depth = fx * tx / (np.abs(u) + 1e-3)
depth[~mask] = np.nan

valid_depth = depth[mask]
min_z, max_z = np.percentile(valid_depth, [2, 98])
depth_norm = (depth - min_z) / (max_z - min_z)
depth_norm = np.clip(depth_norm, 0, 1)
depth_norm[np.isnan(depth_norm)] = 1

# === 提取红蓝区域 ===
B, G, R = img1[:,:,0], img1[:,:,1], img1[:,:,2]
mask_red_color  = (R > 150) & (G < 120) & (B < 120)
mask_blue_color = (B > 150) & (G < 120) & (R < 120)
mask_red_final  = mask_red_color  & mask
mask_blue_final = mask_blue_color & mask

# === 计算平均深度 ===
red_depth  = np.nanmean(depth[mask_red_final])
blue_depth = np.nanmean(depth[mask_blue_final])

print("红块平均深度(数值小=更近):", red_depth)
print("蓝块平均深度(数值小=更近):", blue_depth)
if np.isfinite(red_depth) and np.isfinite(blue_depth):
    rel = (blue_depth - red_depth) / ((red_depth + blue_depth) / 2 + 1e-6) * 100
    who = "红块更近" if red_depth < blue_depth else "蓝块更近"
    print(f"👉 {who}，深度差约 {rel:.1f}%")

# === 仅显示物体区域的深度图 ===
depth_obj = depth_norm.copy()
depth_obj[~(mask_red_final | mask_blue_final)] = np.nan

plt.figure(figsize=(8,6))
plt.imshow(depth_obj, cmap='inferno')
plt.title('Object-relative Depth (white=near, black=far)')
plt.colorbar(label='Normalized depth (over objects)')
plt.axis('off')
plt.show()

# === 可选：叠加掩膜检查 ===
# overlay = img1.copy()
# overlay[mask_red_final]  = [0, 0, 255]
# overlay[mask_blue_final] = [255, 0, 0]
# plt.figure(figsize=(8,6))
# plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
# plt.title('Mask check (red=red block, blue=blue block)')
# plt.axis('off')
# plt.show()