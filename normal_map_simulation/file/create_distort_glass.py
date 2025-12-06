# Generate a distorted plane OBJ with smooth, low-frequency bumps suitable for mirror reflections.
# Coordinates: Z up, X out of screen, Y right; base plane at y≈0, displace along +Y.
import numpy as np

OUT_PATH = "/mnt/data/distorted_plane.obj"

SIZE_X = 2.0       # plane width  (x in [-SIZE_X/2, +SIZE_X/2])
SIZE_Z = 2.0       # plane height (z in [-SIZE_Z/2, +SIZE_Z/2])
GRID   = 320       # subdivision (higher = smoother)
AMP    = 0.055     # displacement amplitude in meters (0.02~0.08 is a good range)

# Build grid
xs = np.linspace(-SIZE_X/2, SIZE_X/2, GRID+1)
zs = np.linspace(-SIZE_Z/2, SIZE_Z/2, GRID+1)
X, Z = np.meshgrid(xs, zs, indexing='ij')

# Smooth, low-freq displacement field Ydisp (sum of gentle sine lobes + a few soft bumps)
# 1) low-frequency sinusoidal base (gives broad "wavy glass")
f1x, f1z = 1.0, 0.8
f2x, f2z = 0.35, 0.27
Ydisp = (
    0.55*np.sin(2*np.pi*(f1x*X/SIZE_X)) * np.cos(2*np.pi*(f1z*Z/SIZE_Z)) +
    0.45*np.sin(2*np.pi*(f2x*X/SIZE_X + f2z*Z/SIZE_Z))
)

# 2) add a few raised-cosine bumps/indentations (positive = bulge, negative = dent)
def soft_bump(X, Z, x0, z0, radius, amp):
    r = np.sqrt((X-x0)**2 + (Z-z0)**2)
    t = np.clip(1 - r/radius, 0, 1)
    w = 0.5 - 0.5*np.cos(np.pi*t)
    return amp*w

bumps = [
    ( +0.60, +0.10, 0.55, +1.0),
    (  0.00,  0.00, 0.70, +0.6),
    ( -0.55, -0.10, 0.45, -0.7),
]
for (x0, z0, rad, amp) in bumps:
    Ydisp += soft_bump(X, Z, x0, z0, rad, amp)

# Normalize to [-1,1] and scale to physical amplitude
Ydisp = Ydisp - Ydisp.mean()
Ydisp /= (np.max(np.abs(Ydisp)) + 1e-9)
Y = AMP * Ydisp  # final displacement along +Y

# Optional Laplacian smoothing to remove sharp transitions
def laplacian_smooth(Y, lam=0.18, iters=3):
    Y2 = Y.copy()
    for _ in range(iters):
        up    = np.roll(Y2, -1, axis=0)
        down  = np.roll(Y2,  1, axis=0)
        left  = np.roll(Y2,  1, axis=1)
        right = np.roll(Y2, -1, axis=1)
        Y2 = (1 - lam) * Y2 + lam * 0.25 * (up + down + left + right)
    return Y2

Y = laplacian_smooth(Y, lam=0.16, iters=3)

# Write OBJ (omit normals; let renderer compute smooth shading)
def idx(i, j):
    return i*(GRID+1) + j + 1

with open(OUT_PATH, "w") as f:
    f.write("# distorted_plane.obj (smooth wavy glass)\n")
    # vertices
    for i in range(GRID+1):
        for j in range(GRID+1):
            x = float(X[i, j])
            y = float(Y[i, j])          # displace along +Y
            z = float(Z[i, j])
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
    # UVs (0..1 from x,z extents)
    xmin, xmax = X.min(), X.max()
    zmin, zmax = Z.min(), Z.max()
    for i in range(GRID+1):
        for j in range(GRID+1):
            u = (X[i, j] - xmin) / (xmax - xmin + 1e-9)
            v = (Z[i, j] - zmin) / (zmax - zmin + 1e-9)
            f.write(f"vt {u:.6f} {v:.6f}\n")
    f.write("s 1\n")
    # faces (two triangles per quad), vt index == v index
    for i in range(GRID):
        for j in range(GRID):
            v00 = idx(i, j);     v10 = idx(i+1, j)
            v01 = idx(i, j+1);   v11 = idx(i+1, j+1)
            f.write(f"f {v00}/{v00} {v10}/{v10} {v11}/{v11}\n")
            f.write(f"f {v00}/{v00} {v11}/{v11} {v01}/{v01}\n")

OUT_PATH
