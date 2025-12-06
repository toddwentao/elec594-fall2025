"""
Compare estimated normals (from recon_out/normals_est.npy) with ground-truth
normals derived from the distorted plane geometry used in the render.

Steps:
 1. Reconstruct the original displacement field Y(x,z) on the 321x321 grid.
 2. Compute analytical normals: n_gt = normalize([-dY/dx, 1, -dY/dz]).
 3. Interpolate GT normals onto the 512x512 rectified grid used in reconstruction.
 4. Load estimated normals and compute per-pixel angular error statistics.
 5. Save error heatmap and summary report.
"""

import os
import numpy as np
from pathlib import Path
from itertools import permutations, product

# Paths (adjust if needed)
RECON_DIR = Path("recon_out")
EST_NORMALS_PATH = RECON_DIR / "normals_est.npy"
ERROR_PNG_PATH = RECON_DIR / "normal_error_deg.png"
ERROR_STATS_PATH = RECON_DIR / "normal_error_stats.txt"

# Geometry constants (must match create_distort_glass.py and reconstruction settings)
SIZE_X = 2.0
SIZE_Z = 2.0
GRID = 320
AMP = 0.055
Wu = 512  # rectification resolution along X
Hv = 512  # rectification resolution along Z


def build_geometry_displacement(): # create experimental glass normal field
    """Recreate the height field Y(x,z) used for the glass mesh."""
    xs = np.linspace(-SIZE_X/2, SIZE_X/2, GRID+1, dtype=np.float64)
    zs = np.linspace(-SIZE_Z/2, SIZE_Z/2, GRID+1, dtype=np.float64)
    X, Z = np.meshgrid(xs, zs, indexing='ij')

    f1x, f1z = 1.0, 0.8
    f2x, f2z = 0.35, 0.27
    Ydisp = (
        0.55*np.sin(2*np.pi*(f1x*X/SIZE_X)) * np.cos(2*np.pi*(f1z*Z/SIZE_Z)) +
        0.45*np.sin(2*np.pi*(f2x*X/SIZE_X + f2z*Z/SIZE_Z))
    )

    def soft_bump(X, Z, x0, z0, radius, amp):
        r = np.sqrt((X-x0)**2 + (Z-z0)**2)
        t = np.clip(1 - r/radius, 0, 1)
        w = 0.5 - 0.5*np.cos(np.pi*t)
        return amp*w

    bumps = [
        (+0.60, +0.10, 0.55, +1.0),
        ( 0.00,  0.00, 0.70, +0.6),
        (-0.55, -0.10, 0.45, -0.7),
    ]
    for (x0, z0, rad, amp_val) in bumps:
        Ydisp += soft_bump(X, Z, x0, z0, rad, amp_val)

    Ydisp = Ydisp - Ydisp.mean()
    Ydisp /= (np.max(np.abs(Ydisp)) + 1e-9)
    Y = AMP * Ydisp

    # Laplacian smoothing (match create_distort_glass.py)
    def laplacian_smooth(Y, lam=0.16, iters=3):
        Y2 = Y.copy()
        for _ in range(iters):
            up    = np.roll(Y2, -1, axis=0)
            down  = np.roll(Y2,  1, axis=0)
            left  = np.roll(Y2,  1, axis=1)
            right = np.roll(Y2, -1, axis=1)
            Y2 = (1 - lam) * Y2 + lam * 0.25 * (up + down + left + right)
        return Y2

    Y = laplacian_smooth(Y, lam=0.16, iters=3)
    return X, Z, Y


def compute_normals_small_slope(X, Z, Y):
    """
    Compute ground-truth normals for a height field y = Y(x, z) where the
    surface normal points roughly along +Y. Small-slope parameterization:
        p = ∂Y/∂x, q = ∂Y/∂z,  n ≈ normalize([-p, 1, -q]).
    """
    dY_dx, dY_dz = np.gradient(Y, X[:,0], Z[0,:], edge_order=2)
    n = np.stack([-dY_dx, np.ones_like(dY_dx), -dY_dz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return n


def interpolate_normals_to_grid(X, Z, n, Wu, Hv):
    """Interpolate GT normals onto reconstruction grid (Wu x Hv)."""
    x_target = np.linspace(X.min(), X.max(), Wu, dtype=np.float64)
    z_target = np.linspace(Z.min(), Z.max(), Hv, dtype=np.float64)

    # Flatten source
    n_flat = n.reshape(-1, 3)
    X_flat = X.reshape(-1)
    Z_flat = Z.reshape(-1)

    # Use scipy if available; else fall back to manual bilinear interpolation
    try:
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (X[:,0], Z[0,:]),
            n,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        Xt, Zt = np.meshgrid(x_target, z_target, indexing='xy')
        pts = np.stack([Xt, Zt], axis=-1)
        n_interp = interp((pts[...,0], pts[...,1]))
    except ImportError:
        # Manual bilinear interpolation
        n_interp = np.zeros((Hv, Wu, 3), dtype=np.float64)
        for c in range(3):
            n_interp[...,c] = bilinear_interpolate(
                X, Z, n[...,c], x_target, z_target
            )
    return n_interp


def bilinear_interpolate(X, Z, values, x_target, z_target):
    """Manual bilinear interpolation over a rect grid."""
    # Compute grid spacing (uniform)
    dx = X[1,0] - X[0,0]
    dz = Z[0,1] - Z[0,0]

    x0 = X.min()
    z0 = Z.min()

    Xi = ((x_target - x0) / dx).clip(0, X.shape[0]-1.00001)
    Zi = ((z_target - z0) / dz).clip(0, Z.shape[1]-1.00001)

    xi0 = np.floor(Xi).astype(int)
    zi0 = np.floor(Zi).astype(int)
    xi1 = np.clip(xi0 + 1, 0, X.shape[0]-1)
    zi1 = np.clip(zi0 + 1, 0, Z.shape[1]-1)

    wx = Xi - xi0
    wz = Zi - zi0

    out = np.zeros((len(z_target), len(x_target)), dtype=np.float64)
    for i, zz in enumerate(z_target):
        z00 = zi0[i]
        z11 = zi1[i]
        wz_i = wz[i]
        for j, xx in enumerate(x_target):
            x00 = xi0[j]
            x11 = xi1[j]
            wx_j = wx[j]

            v00 = values[x00, z00]
            v10 = values[x11, z00]
            v01 = values[x00, z11]
            v11 = values[x11, z11]

            out[i, j] = (
                (1-wx_j)*(1-wz_i)*v00 +
                wx_j*(1-wz_i)*v10 +
                (1-wx_j)*wz_i*v01 +
                wx_j*wz_i*v11
            )
    return out


def compute_angle_errors(n_est, n_gt):
    dot = np.clip(np.sum(n_est * n_gt, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def evaluate_axis_permutations(n_est, n_gt):
    """
    Try all permutations and sign flips of estimated normals to find the best
    alignment with ground truth. Returns (best_normals, best_perm, best_signs, stats).
    """
    best = None
    best_stats = None
    best_perm = None
    best_signs = None

    for perm in permutations(range(3)):
        permuted = n_est[..., perm]
        for signs in product([1, -1], repeat=3):
            candidate = permuted * np.array(signs, dtype=np.float64)
            candidate /= np.linalg.norm(candidate, axis=-1, keepdims=True) + 1e-12
            err = compute_angle_errors(candidate, n_gt)
            mean_err = float(np.nanmean(err))
            median_err = float(np.nanmedian(err))
            max_err = float(np.nanmax(err))
            if best is None or mean_err < best_stats[0]:
                best = candidate
                best_stats = (mean_err, median_err, max_err)
                best_perm = perm
                best_signs = signs
    return best, best_perm, best_signs, best_stats


def main():
    assert EST_NORMALS_PATH.exists(), f"Estimated normals not found: {EST_NORMALS_PATH}"

    print("Reconstructing ground-truth height field...")
    X, Z, Y = build_geometry_displacement()

    print("Computing ground-truth normals (small-slope parameterization)...")
    n_gt_full = compute_normals_small_slope(X, Z, Y)

    print("Interpolating GT normals to reconstruction grid...")
    n_gt = interpolate_normals_to_grid(X, Z, n_gt_full, Wu=Wu, Hv=Hv)
    n_gt /= np.linalg.norm(n_gt, axis=-1, keepdims=True) + 1e-12

    print("Loading estimated normals...")
    n_est = np.load(EST_NORMALS_PATH)
    assert n_est.shape == (Hv, Wu, 3), f"Estimated normals shape mismatch: {n_est.shape}"

    print("Searching best axis/sign alignment...")
    n_aligned, best_perm, best_signs, stats = evaluate_axis_permutations(n_est, n_gt)
    mean_err, median_err, max_err = stats

    print(f"Best permutation: {best_perm}, signs: {best_signs}")
    print(f"Mean error:   {mean_err:.3f}°")
    print(f"Median error: {median_err:.3f}°")
    print(f"Max error:    {max_err:.3f}°")

    # Save stats
    with open(ERROR_STATS_PATH, "w") as f:
        f.write(f"Mean error (deg):   {mean_err:.6f}\n")
        f.write(f"Median error (deg): {median_err:.6f}\n")
        f.write(f"Max error (deg):    {max_err:.6f}\n")
        f.write(f"Best permutation: {best_perm}\n")
        f.write(f"Best signs: {best_signs}\n")

    # Save heatmap (optional)
    try:
        import imageio.v3 as iio
        error_deg = compute_angle_errors(n_aligned, n_gt)
        vmax = np.percentile(error_deg, 99)
        heat = np.clip(error_deg / (vmax + 1e-6), 0, 1)
        heat_img = (plt_colormap(heat) * 255).astype(np.uint8)
        iio.imwrite(ERROR_PNG_PATH, heat_img)
        print(f"Saved error heatmap to {ERROR_PNG_PATH}")
    except ImportError:
        print("imageio not available; skipping heatmap generation.")

    print(f"Saved stats to {ERROR_STATS_PATH}")


def plt_colormap(norm_vals):
    """Simple viridis-like colormap using numpy (no matplotlib dependency)."""
    # Coefficients adapted from matplotlib's viridis
    x = norm_vals[..., None]
    c = np.concatenate([
        0.280268003 + 0.391672101*x + 1.57870059*x**2 - 2.53466052*x**3,
        0.165995648 + 1.02163186*x - 0.53311624*x**2 - 1.66983027*x**3,
        0.476530682 - 0.01818914*x - 1.57770082*x**2 + 3.58083148*x**3
    ], axis=-1)
    return np.clip(c, 0, 1)


if __name__ == "__main__":
    main()
