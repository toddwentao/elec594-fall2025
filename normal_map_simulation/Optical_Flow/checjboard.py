# import os
# import numpy as np
# import cv2

# # 输出路径：生成的棋盘纹理会保存在这里
# output_path = '/Users/young/capstone_3D/Optical_Flow/textures/my_checker.png'

# # 确保目录存在
# os.makedirs(os.path.dirname(output_path), exist_ok=True)

# # 定义棋盘大小和格子数
# img_size = 512        # 图像尺寸为 512×512 像素
# num_tiles = 8         # 8×8 个格子
# tile_size = img_size // num_tiles

# # 初始化空图像
# checkerboard = np.zeros((img_size, img_size, 3), dtype=np.uint8)

# # 生成棋盘格子，交替填充颜色
# for row in range(num_tiles):
#     for col in range(num_tiles):
#         # 根据行列索引决定颜色，偶数格为浅色，奇数格为深色
#         color = (230, 230, 230) if (row + col) % 2 == 0 else (50, 50, 50)
#         y_start = row * tile_size
#         y_end   = y_start + tile_size
#         x_start = col * tile_size
#         x_end   = x_start + tile_size
#         checkerboard[y_start:y_end, x_start:x_end] = color

# # 保存为 PNG 文件
# cv2.imwrite(output_path, checkerboard)
# print(f'棋盘纹理已生成: {output_path}')

import os
import numpy as np
import cv2

# 生成棋盘格的函数
def generate_checker(color_base, save_path, img_size=512, num_tiles=8):
    """
    color_base: RGB 归一化基色，例如 [0.8, 0.1, 0.1]
    save_path: 保存文件的完整路径
    img_size: 图片尺寸（像素）
    num_tiles: 棋盘格的格子数量（每边）
    """
    tile_size = img_size // num_tiles
    board = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # 构建棋盘格：偶数格用基色，奇数格用稍亮的基色
    for i in range(num_tiles):
        for j in range(num_tiles):
            # 根据格子坐标决定颜色
            if (i + j) % 2 == 0:
                color = (np.array(color_base) * 255).astype(np.uint8)
            else:
                # 适当调亮基色，以增加对比度
                color = (np.array(color_base) * 255 * 0.6 + 255 * 0.4).astype(np.uint8)
            board[
                i * tile_size : (i + 1) * tile_size,
                j * tile_size : (j + 1) * tile_size,
            ] = color

    # 保存 PNG 文件
    cv2.imwrite(save_path, board)
    print(f"已生成：{save_path}")

# 设置保存目录
texture_dir = "/Users/young/capstone_3D/Optical_Flow/textures"
os.makedirs(texture_dir, exist_ok=True)

# 生成红色和蓝色棋盘格
generate_checker([0.8, 0.1, 0.1], os.path.join(texture_dir, "checker_red.png"))
generate_checker([0.1, 0.1, 0.8], os.path.join(texture_dir, "checker_blue.png"))