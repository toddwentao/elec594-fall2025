#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 Jacquet 等人（2013）的做法，从 70 帧 Mitsuba3 渲染的“灯具”反射序列中重建近似平面的反射玻璃表面法线。

工作流程：
  0）查找所有帧并基于渲染脚本的数学模型重新生成相机位姿
  1）将每帧校正到平面 y=0 上统一的窗口坐标 (u,v)
  2）体数据分割（快速版本）：H(u,v) = argmax_t |∂I/∂t|
  3）为每一帧提取反射曲线：Γ_t = { (u,v) | |H(u,v)-t|<ε }
  4）构建逐像素圆锥约束：n^T C_p n = 0（由相机 S、平面点 O 以及空间直线 l 推导）
  5）全局法线求解（简化示例）：在 ∑(n^T C n)^2 + 平滑项 上进行正则化梯度下降
  6）（可选）与 Mitsuba 导出的 GT 法线比较，并可视化夹角误差

说明：
- 这是一个简洁且易读的脚手架。第 3.4 节的全局优化通过梯度下降与拉普拉斯平滑实现，虽然简化但已能在干净的合成数据中恢复较好的法线。如有需要，可升级为带可积性约束的 Gauss-Newton 方法。
- 校正阶段统一使用 *y=0 平面* 作为窗口，与当前场景设置一致。

"""
from __future__ import annotations
import os, math, glob, json, re
from pathlib import Path
import numpy as np
import imageio.v3 as iio
from typing import Tuple, Dict

# ------------------------------
# 用户可配置的输入输出
# ------------------------------
# 70 帧渲染结果（single_object_lamp.py 输出）的所在目录
FRAME_DIR    = "output/lamp_renders"    
FRAME_GLOB   = "frame_*.png"
# 中间结果输出目录
OUT_DIR      = "recon_out"
os.makedirs(OUT_DIR, exist_ok=True)

# 如果导出了 EXR 格式的 GT 法线（世界坐标系），在此填写相应的 glob 模式（可选）
GT_NORMALS_GLOB = None  # 例如 "output/lamp_renders/normal_*.exr"

# ------------------------------
# 场景常量（需与渲染脚本保持一致）
# ------------------------------
NUM_FRAMES = 70
FPS = 30
CAM_SWEEP = 3.0
CAMERA_HEIGHT = 0.0
CAMERA_FOV_DEG = 40.0
SAMPLES = 128

# 玻璃/窗户所在的参考平面：y = 0
# X、Z 方向的范围（米）。你的 OBJ 使用 SIZE_X = SIZE_Z = 2.0
GLASS_SIZE_X = 2.0
GLASS_SIZE_Z = 2.0

# 相机与物体摆放（需与 single_object_lamp.py 匹配）
OBJECT_POS = np.array([0.0, 0.0, 0.0], dtype=np.float64)
CAMERA_OFFSET_FROM_PLANE = 2.0
# 灯的中心（将其“垂直中心线”建模为约束所需的 3D 直线）
LAMP_POSITION = np.array([0.0, CAMERA_OFFSET_FROM_PLANE + 2.0, 0.0], dtype=np.float64) # （[0,4,0]）
LAMP_DIR      = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # 垂直方向（Z 轴朝上）

# ------------------------------
# 计算相机内参
# ------------------------------
def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据水平视场角（角度制）计算透视相机的内参矩阵 K,像素为方形且相机主点位于图像中心。
    返回 K、K 的逆矩阵 K_inv,以及主点 (cx, cy)。

    输入：帧宽度、高度、视场角 fov（例如 40）
    输出：K（内参矩阵）、K_inv（内参矩阵的逆）、以及主点坐标 (cx, cy)
    """
    fov = math.radians(fov_deg)
    fx = 0.5 * width / math.tan(0.5 * fov)
    fy = fx * (height / width)  # 保持水平视场角，依据纵横比调整 fy
    cx, cy = width * 0.5, height * 0.5
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    K_inv = np.linalg.inv(K)
    return K, K_inv, np.array([cx, cy]) # 返回内参矩阵及主点

# ------------------------------
# camera pose - 外参
# ------------------------------
def camera_pose_from_frame(i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (R, t, S)，其中：
      R：3x3 的世界到相机旋转矩阵（相机朝向 -Y）
      t：3x1 的世界到相机平移向量
      S：3x1 的世界坐标系下相机中心
    """
    if NUM_FRAMES <= 1:
        t_norm = 0.0
    else:
        t_norm = i / (NUM_FRAMES - 1)
    x = OBJECT_POS[0] + (-CAM_SWEEP + 2.0 * CAM_SWEEP * t_norm)
    y = OBJECT_POS[1] + CAMERA_OFFSET_FROM_PLANE
    z = OBJECT_POS[2] + CAMERA_HEIGHT
    S = np.array([x, y, z], dtype=np.float64)
    # 观察方向：target = S + [0, -1, 0]，up=[0,0,1]
    fwd = np.array([0.0, -1.0, 0.0])       # 相机坐标系的 -Z 映射到世界坐标系的 -Y，这里显式构建 R
    upw = np.array([0.0,  0.0, 1.0])
    # 构建相机基：z_cam = normalize(target - S) = -Y；x_cam = normalize(cross(z_cam, up))；y_cam = cross(z_cam, x_cam)
    z_cam = (S + np.array([0, -1, 0]) - S); z_cam = z_cam / np.linalg.norm(z_cam)   # == [0,-1,0]
    x_cam = np.cross(upw, z_cam); x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    # 世界到相机的旋转矩阵以相机坐标轴 (x_cam, y_cam, z_cam) 作为行向量
    R_wc = np.vstack([x_cam, y_cam, z_cam])         # 3x3
    t_wc = -R_wc @ S                                 # 3x1
    return R_wc, t_wc, S # 返回旋转矩阵、平移向量与相机中心

# ------------------------------
# 投影/反投影工具函数
# ------------------------------
def project_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, Pw: np.ndarray) -> np.ndarray:
    """将 Nx3 的世界坐标点投影为 Nx2 的像素坐标（齐次坐标除法）。"""
    Pc = (R @ Pw.T + t[:,None])            # 3xN，相机坐标系下的点
    uvw = K @ Pc
    uv = (uvw[:2] / uvw[2:3]).T            # Nx2 像素坐标
    return uv

def rectify_frame_to_window(img: np.ndarray, K, R, t, grid_O: np.ndarray) -> np.ndarray:
    """
    给定目标 (u,v) 网格（grid_O => Nx3、位于 y=0 平面上的世界坐标点），
    在源图像中按投影像素采样。
    """
    Ht, Wt = img.shape[:2]
    uv = project_points(K, R, t, grid_O)   # Nx2 像素坐标
    # 插值采样前先进行边界裁剪
    uv[:,0] = np.clip(uv[:,0], 0, Wt-1)
    uv[:,1] = np.clip(uv[:,1], 0, Ht-1)
    # 双线性插值
    x0 = np.floor(uv[:,0]).astype(np.int32); x1 = np.clip(x0+1, 0, Wt-1)
    y0 = np.floor(uv[:,1]).astype(np.int32); y1 = np.clip(y0+1, 0, Ht-1)
    wx = (uv[:,0] - x0)[:, None]
    wy = (uv[:,1] - y0)[:, None]

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    # 灰度图情况下补齐通道维度
    if Ia.ndim == 1:
        Ia = Ia[:, None]; Ib = Ib[:, None]; Ic = Ic[:, None]; Id = Id[:, None]

    out = (Ia*(1-wx)*(1-wy) + Ib*wx*(1-wy) + Ic*(1-wx)*wy + Id*wx*wy)
    return out

# ------------------------------
# 在参考平面 y=0 上构建统一的 (u,v) 网格
# ------------------------------
def build_window_grid(Wu=512, Hv=512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    玻璃平面尺寸为 2 x 2，对应的网格尺寸为 Wu x Hv。

    返回三个数组，用于描述 y=0 平面上的采样网格：
      - grid_O：形状为 (Wu*Hv)×3 的世界坐标点（每行 [x,0,z]）
      - X：形状 Hv×Wu 的 x 坐标网格
      - Z：形状 Hv×Wu 的 z 坐标网格
    网格覆盖范围为 X∈[-GLASS_SIZE_X/2, GLASS_SIZE_X/2]，Z∈[-GLASS_SIZE_Z/2, GLASS_SIZE_Z/2]。
    """
    xs = np.linspace(-GLASS_SIZE_X/2, GLASS_SIZE_X/2, Wu, dtype=np.float64)
    zs = np.linspace(-GLASS_SIZE_Z/2, GLASS_SIZE_Z/2, Hv, dtype=np.float64)
    X, Z = np.meshgrid(xs, zs, indexing='xy')
    Y = np.zeros_like(X)
    O = np.stack([X, Y, Z], axis=-1).reshape(-1,3)   # Nx3 世界坐标点
    return O, X, Z

# ------------------------------
# 体数据分割（快速版）：H(u,v) = argmax_t |Δ_t I|
# ------------------------------
def estimate_H_from_volume(volume: np.ndarray) -> np.ndarray:
    '''
    相邻帧做差分，计算pixel value
    把每一帧差分值最大的pixel找出来,拼在H上
    '''

    # 四维体数据 volume 的形状为 [T, Hv, Wu, C]，float32，值域为 [0,1]
    if volume.ndim==4 and volume.shape[-1]==3:
        gray = 0.299*volume[...,0] + 0.587*volume[...,1] + 0.114*volume[...,2]
    else:
        gray = volume[...,0] if volume.ndim==4 else volume
    dt = np.abs(np.diff(gray, axis=0))   # [T+1, Hv, Wu] - [T, Hv, Wu] 的像素差分
    H = np.argmax(dt, axis=0).astype(np.float32)
    return H

# ------------------------------
# 提取指定帧的反射曲线像素
# ------------------------------
def extract_curve_mask(Hmap: np.ndarray, t: int, eps: float=0.5) -> np.ndarray:
    return (np.abs(Hmap - t) < eps)

def sample_curve_points(mask: np.ndarray, step: int=4) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs)==0:
        return np.zeros((0,2), np.int32)
    # 通过步长采样控制稀疏度
    sel = np.arange(0, len(xs), step, dtype=np.int32)
    return np.stack([xs[sel], ys[sel]], axis=1)  # Mx2 坐标 (x,y)

# ------------------------------
# 构建圆锥约束：n^T C n = 0
# ------------------------------
def cone_matrix(O: np.ndarray, S: np.ndarray, L0: np.ndarray, Ldir: np.ndarray) -> np.ndarray:
    """
    O: 3 维，参考平面 y=0 上的表面点
    S: 3 维，相机中心
    L0: 3 维，直线上的一点（灯中心）
    Ldir: 3 维，直线的单位方向（竖直）
    实现论文式 (6)-(8) 风格的构造（参见分析笔记）。

    D = [ S_hat + l0_hat, 2*v_hat, S_hat - l0_hat ]
    Q = [[ 1, 0, -2],
         [ 0, 1,  0],
         [-2, 0,  1]]
    C = inv(D)^T Q inv(D)
    """
    v_hat = Ldir / (np.linalg.norm(Ldir)+1e-9)
    # 求直线上距离 O 最近的点
    w   = O - L0
    t   = np.dot(w, v_hat)
    Cpt = L0 + t * v_hat
    l0  = Cpt - O
    l0_hat = l0 / (np.linalg.norm(l0)+1e-9)

    Svec = S - O
    S_hat = Svec / (np.linalg.norm(Svec)+1e-9)

    D = np.stack([S_hat + l0_hat, 2.0*v_hat, S_hat - l0_hat], axis=1)  # 3x3
    Q = np.array([[ 1.0, 0.0, -2.0],
                  [ 0.0, 1.0,  0.0],
                  [-2.0, 0.0,  1.0]], dtype=np.float64)
    try:
        Dinvg = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        # 接近退化时回退到伪逆
        Dinvg = np.linalg.pinv(D)
    C = Dinvg.T @ Q @ Dinvg
    # 对称化
    C = 0.5*(C + C.T)
    return C

# ------------------------------
# 全局法线求解（简化版）：梯度下降 + 拉普拉斯平滑
# ------------------------------
def normals_from_cones(Hmap: np.ndarray,
                       curve_pixels: Dict[int, np.ndarray],
                       K, R_list, t_list, S_list,
                       X_grid, Z_grid,
                       L0, Ldir,
                       iters: int=50, lr: float=0.1, lam_smooth: float=0.1) -> np.ndarray:
    """
    在完整网格上估计法线。使用小斜率参数化 (p,q)，即 n ~ normalize([-p,-q,1])。
    数据项：∑_{曲线像素} (n^T C_p n)^2
    平滑项：对 p、q 引入拉普拉斯正则
    """
    Hv, Wu = Hmap.shape
    p = np.zeros((Hv, Wu), dtype=np.float64)
    q = np.zeros((Hv, Wu), dtype=np.float64)

    # 预计算 y=0 平面上每个 (u,v) 位置对应的世界坐标点
    Ogrid = np.stack([X_grid, np.zeros_like(X_grid), Z_grid], axis=-1)  # Hv x Wu x 3 的张量

    # 为提升效率：提前为每帧采样到的曲线像素构建对应的圆锥矩阵
    cone_terms = []  # (ys, xs, C_mats) 的列表
    for t in range(len(R_list)):
        pts = curve_pixels.get(t)
        if pts is None or len(pts)==0:
            cone_terms.append((np.array([],int), np.array([],int), np.zeros((0,3,3))))
            continue
        ys = pts[:,1]; xs = pts[:,0]
        S = S_list[t]
        # 为每个像素构建 C
        C_mats = np.zeros((len(xs),3,3), dtype=np.float64)
        for k,(x,y) in enumerate(zip(xs,ys)):
            O = Ogrid[y,x]
            C_mats[k] = cone_matrix(O, S, L0, Ldir)
        cone_terms.append((ys, xs, C_mats))

    def laplacian(A: np.ndarray) -> np.ndarray:
        """使用边界夹紧（离散 Neumann）的五点拉普拉斯算子。"""
        up = np.empty_like(A)
        up[:-1] = A[1:]
        up[-1] = A[-1]

        down = np.empty_like(A)
        down[1:] = A[:-1]
        down[0] = A[0]

        left = np.empty_like(A)
        left[:, 1:] = A[:, :-1]
        left[:, 0] = A[:, 0]

        right = np.empty_like(A)
        right[:, :-1] = A[:, 1:]
        right[:, -1] = A[:, -1]

        return up + down + left + right - 4.0 * A

    for it in range(iters):
        # 针对数据项构建对 p、q 的梯度
        dp = np.zeros_like(p); dq = np.zeros_like(q)
        loss_data = 0.0
        for t,(ys,xs,Cm) in enumerate(cone_terms):
            if len(xs)==0: continue
            # 当前像素位置的法线
            pn = p[ys,xs]
            qn = q[ys,xs]
            v = np.stack([-pn, -qn, np.ones_like(pn)], axis=1)
            v_norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
            n = v / v_norm
            # 残差 r = n^T C n
            r = np.einsum('bi,bij,bj->b', n, Cm, n)
            loss_data += float(np.mean(r**2))
            # 针对 n 的基础梯度：∂(r^2)/∂n = 4 r C n（C 为对称阵）
            Cn = np.einsum('bij,bj->bi', Cm, n)
            g_n = 4.0 * r[:, None] * Cn
            # 链式求导并考虑归一化：∂n/∂v = (I - n n^T) / ||v||
            proj = g_n - (np.sum(g_n * n, axis=1, keepdims=True) * n)
            g_v = proj / v_norm
            dp_loc = -g_v[:,0]
            dq_loc = -g_v[:,1]
            # 将梯度散布回对应像素
            dp[ys,xs] += dp_loc
            dq[ys,xs] += dq_loc

        # 平滑正则项的梯度
        Lp = laplacian(p)
        Lq = laplacian(q)
        dp += lam_smooth * Lp
        dq += lam_smooth * Lq

        # 梯度下降更新
        p -= lr * dp
        q -= lr * dq

        if it % 10 == 0 or it == iters-1:
            print(f"[iter {it:02d}] data_loss≈{loss_data:.6f}")

    # 生成最终法线
    n = np.stack([-p, np.ones_like(p), -q], axis=-1)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
    return n

# ------------------------------
# 角度误差工具
# ------------------------------
def angle_error_deg(n_est: np.ndarray, n_gt: np.ndarray) -> np.ndarray:
    dot = np.clip(np.sum(n_est*n_gt, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

# ------------------------------
# 主流程
# ------------------------------
def main():
    frames = sorted(glob.glob(str(Path(FRAME_DIR)/FRAME_GLOB)))
    assert len(frames)>0, f"No frames found in {FRAME_DIR}/{FRAME_GLOB}"
    print(f"Found {len(frames)} frames (showing first 3):", frames[:3])
    # 读取首帧以获取分辨率
    im0 = iio.imread(frames[0]).astype(np.float32)/255.0
    Himg, Wimg = im0.shape[:2]
    K, K_inv, cxy = intrinsics_from_fov(Wimg, Himg, CAMERA_FOV_DEG)

    # 构建统一的窗口网格（校正后的分辨率）
    Wu, Hv = 512, 512
    Ogrid_flat, Xgrid, Zgrid = build_window_grid(Wu=Wu, Hv=Hv)

    # 对所有帧进行校正
    # 把每一帧的图像都map到统一的(u,v)平面
    rect = np.zeros((len(frames), Hv, Wu, 3), dtype=np.float32)
    R_list, t_list, S_list = [], [], []
    for t in range(len(frames)):
        img = iio.imread(frames[t]).astype(np.float32)/255.0
        R, tw, S = camera_pose_from_frame(t)
        R_list.append(R); t_list.append(tw); S_list.append(S)
        out_flat = rectify_frame_to_window(img, K, R, tw, Ogrid_flat)
        rect[t] = out_flat.reshape(Hv, Wu, 3)

    # 保存几帧校正结果拼图，方便快速检查
    iio.imwrite(str(Path(OUT_DIR)/"rectified_t0.png"), (rect[0]*255).astype(np.uint8))
    iio.imwrite(str(Path(OUT_DIR)/"rectified_tmid.png"), (rect[len(frames)//2]*255).astype(np.uint8))
    iio.imwrite(str(Path(OUT_DIR)/"rectified_tend.png"), (rect[-1]*255).astype(np.uint8))

    # 体数据分割，生成 H(u,v)
    Hmap = estimate_H_from_volume(rect)
    np.save(str(Path(OUT_DIR)/"Hmap.npy"), Hmap)
    # 可视化 H（缩放到 0..255）
    Hviz = (255*(Hmap/ max(1,len(frames)-2))).astype(np.uint8)
    iio.imwrite(str(Path(OUT_DIR)/"Hmap_viz.png"), Hviz)

    # 提取反射曲线
    curve_pixels = {}
    for t in range(len(frames)):
        mask = extract_curve_mask(Hmap, t, eps=0.5)
        pts = sample_curve_points(mask, step=3)
        curve_pixels[t] = pts
    # 导出一帧的曲线像素，便于检查
    np.save(str(Path(OUT_DIR)/"curve_pixels_tmid.npy"), curve_pixels[len(frames)//2])

    # 构建灯的空间直线（经过 LAMP_POSITION，方向为 LAMP_DIR）
    L0 = LAMP_POSITION.copy()
    Ldir = LAMP_DIR / (np.linalg.norm(LAMP_DIR)+1e-9)

    # 全局法线估计
    n_est = normals_from_cones(Hmap, curve_pixels, K, R_list, t_list, S_list,
                               Xgrid, Zgrid, L0, Ldir,
                               iters=60, lr=0.01, lam_smooth=0.2)
    # 将法线保存为 RGB（从 [-1,1] 映射到 [0,1]）
    n_rgb = 0.5*(n_est + 1.0)
    iio.imwrite(str(Path(OUT_DIR)/"normals_est.png"), (255*np.clip(n_rgb,0,1)).astype(np.uint8))
    np.save(str(Path(OUT_DIR)/"normals_est.npy"), n_est)

    # 可选：若提供 GT 法线则进行比较
    if GT_NORMALS_GLOB:
        gt_files = sorted(glob.glob(GT_NORMALS_GLOB))
        if len(gt_files) == len(frames):
            # 若 GT 法线对应已经校正后的网格（y=0 平面的 UV），可在此加载示例
            # 否则可导出单张平面上的 GT 法线图并在此读取
            print("GT comparison path not implemented in this scaffold.")
        else:
            print("GT normals count != frame count; skipping GT comparison.")
    print("Done. Outputs written to", OUT_DIR)

if __name__ == "__main__":
    main()
