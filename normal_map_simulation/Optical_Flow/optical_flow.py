import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取两帧
img1 = cv2.imread('Optical_Flow/mirror_scene/images/frame_000.png')
img2 = cv2.imread('Optical_Flow/mirror_scene/images/frame_001.png')

# 转灰度
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 计算光流
flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                    pyr_scale=0.5, levels=3, winsize=15,
                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# 光流分量
fx, fy = flow[...,0], flow[...,1]
mag, ang = cv2.cartToPolar(fx, fy)

# 可视化水平光流
plt.imshow(fx, cmap='jet')
plt.title('Horizontal optical flow (u)')
plt.colorbar()
plt.show()

mask_red = (img1[:,:,2] > 100) & (img1[:,:,0] < 50)
mask_blue = (img1[:,:,0] > 100) & (img1[:,:,2] < 50)

mean_u_red = np.mean(np.abs(fx[mask_red]))
mean_u_blue = np.mean(np.abs(fx[mask_blue]))

print(f"红块光流均值: {mean_u_red:.2f}, 蓝块光流均值: {mean_u_blue:.2f}")
if mean_u_red > mean_u_blue:
    print("→ 红色柱体更近")
else:
    print("→ 蓝色柱体更近")