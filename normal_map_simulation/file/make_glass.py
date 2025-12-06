# make_gentle_bent_glass.py
# 生成一块“微微弯曲”的玻璃 OBJ：可选单轴圆柱弯曲或双轴轻微弯曲
# 坐标：Z UP，X out ,right；位移沿 +Y（面朝相机）

import math
import numpy as np

# -------- 可调参数（建议先用默认） --------
SIZE_X = 10       # 宽度（x ∈ [-SIZE_X/2, +SIZE_X/2]）
SIZE_Z = 10       # 高度（z ∈ [-SIZE_Z/2, +SIZE_Z/2]）
GRID   = 360       # 网格细分（越大越顺滑，200~360都行）

# 圆柱/双轴弯曲半径（米）。半径越大越“更微”，越小越“更弯”
R_X = 10.0         # 沿 X 方向的弯曲半径（圆柱面）   e.g. 6~12
R_Z = None        # 沿 Z 方向的弯曲半径（可选）。None=只沿X弯；设如 10.0 可做轻微双轴

# 强度（0~1）。用于微调弯曲量；一般 0.3~1.0
STRENGTH_X = 0.9
STRENGTH_Z = 1.0

OUT_PATH = "scenes/my_meshes/glass.obj"
# ----------------------------------------

def bend_y_cyl(x, R, strength=1.0):
    """单轴圆柱弯曲：y = R*(1 - cos(x/R))，x=0处为0；strength用于线性缩放"""
    if (R is None) or (R <= 0) or (strength == 0):
        return 0.0
    return strength * (R * (1.0 - math.cos(x / R)))

def main():
    xs = np.linspace(-SIZE_X/2, SIZE_X/2, GRID+1)
    zs = np.linspace(-SIZE_Z/2, SIZE_Z/2, GRID+1)
    X, Z = np.meshgrid(xs, zs, indexing='ij')  # (GRID+1, GRID+1)

    # 位移沿 +Y：单轴或双轴叠加
    Y = np.zeros_like(X, dtype=np.float32)
    # 用二阶近似也行 y≈x^2/(2R)，但 cos 形式在大位移时更稳定
    f_bend_x = np.vectorize(lambda x: bend_y_cyl(x, R_X, STRENGTH_X))
    Y += f_bend_x(X)
    if R_Z is not None:
        f_bend_z = np.vectorize(lambda z: bend_y_cyl(z, R_Z, STRENGTH_Z))
        Y += f_bend_z(Z)

    # 零均值（不整体抬升/下压）
    Y -= np.mean(Y)

    # 写 OBJ（不写法线，让渲染器做平滑法线）
    verts = []
    uvs = []
    xmin, xmax = X.min(), X.max()
    zmin, zmax = Z.min(), Z.max()
    for i in range(GRID+1):
        for j in range(GRID+1):
            x, y, z = X[i, j], float(Y[i, j]), Z[i, j]
            verts.append((x, y, z))
            u = (x - xmin) / (xmax - xmin + 1e-9)
            v = (z - zmin) / (zmax - zmin + 1e-9)
            uvs.append((u, v))

    def idx(i, j): return i*(GRID+1) + j + 1

    faces = []
    for i in range(GRID):
        for j in range(GRID):
            v00 = idx(i, j);     v10 = idx(i+1, j)
            v01 = idx(i, j+1);   v11 = idx(i+1, j+1)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    with open(OUT_PATH, "w") as f:
        f.write("# gentle_bent_glass.obj\n")
        for (x,y,z) in verts: f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for (u,v) in uvs:     f.write(f"vt {u:.6f} {v:.6f}\n")
        f.write("s 1\n")
        for (a,b,c) in faces: f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    print(f"✅ Saved: {OUT_PATH}  (GRID={GRID}, R_X={R_X}, R_Z={R_Z})")

if __name__ == "__main__":
    main()
