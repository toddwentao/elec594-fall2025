"""
Scene generation script with fiducial markers.

This module is a drop‐in replacement for the earlier Mitsuba scene script.
It constructs a mirror scene containing two rectangular blocks (one red and
one blue) behind a planar mirror and augments the scene with small,
coloured spheres on the top corners of each block.  These spheres act as
fiducial markers to facilitate feature detection and correspondence
matching between frames when using classical computer vision techniques.

The script renders three frames from slightly different camera yaw angles
and writes both the images and the intrinsic/extrinsic camera parameters
to disk.  It follows the same interface as the original script: running
this file will create a directory ``mirror_scene`` in the ``Optical_Flow``
folder containing the rendered frames, a ``poses.json`` file with the
camera extrinsics and a ``intrinsics.json`` with the camera intrinsics.

To use this script, call ``python create_scene_campara.py`` from the root
of the project.  The resulting markers are brightly coloured and positioned
on the top edges of the red and blue blocks, making them easy to detect
with ORB, SIFT, or other feature detectors.

"""

import os
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import mitsuba as mi

# Ensure we use the scalar RGB variant; this is faster for simple scenes
mi.set_variant("scalar_rgb")

# ---------------------------------------------------------------------------
# Texture configuration
#
# The blocks in this scene use a checkerboard texture to introduce
# high‑frequency details that are useful for optical flow estimation.
# Below we define an absolute path to the texture file based on the
# location of this Python script.  When loading the texture in the scene
# description, the `filename` field should reference this variable.
import os as _os
_SCRIPT_DIR = _os.path.dirname(__file__)
# Paths to checkerboard textures.  To avoid reliance on Mitsuba's ``scale``
# texture plugin (which may not be available in some builds), we pre‑tint
# the checkerboard images in Python and save them alongside this script.  Each
# block uses its own tinted texture file.  See ``generate_tinted_checkerboards``
# in the repository for how these images are created.
CHECKER_TEXTURE_RED = _os.path.join(_SCRIPT_DIR, 'textures', 'checker_red.png')
CHECKER_TEXTURE_BLUE = _os.path.join(_SCRIPT_DIR, 'textures', 'checker_blue.png')

# ========================= Configurable parameters ==========================
# Output directory for frames and metadata
OUT_DIR = "./Optical_Flow/mirror_scene"
# Image resolution
IMG_W, IMG_H = 640, 480
# Horizontal field of view in degrees
FOV_DEG = 45.0
# Camera z coordinate (distance from mirror)
Z0 = 5.0
# Translation offsets for the camera along the x‑axis (units in world metres).
# During rendering, the camera will keep the same orientation as the central
# view but will be shifted left/right by these offsets.  Adjust these values
# to control how far the camera moves parallel to the mirror.
X_OFFSETS = [-0.5, 0.0, 0.5]


def _rotation_matrix_y(theta: float) -> np.ndarray:
    """Return a 3×3 rotation matrix for rotation around the Y axis by theta radians."""
    return np.array([
        [math.cos(theta), 0.0, math.sin(theta)],
        [0.0, 1.0, 0.0],
        [-math.sin(theta), 0.0, math.cos(theta)],
    ])


def compute_marker_positions() -> List[Dict[str, List[float]]]:
    """
    Compute world coordinates and colours for fiducial markers on the blocks.

    The scene contains two rectangular blocks scaled from the unit cube.  We
    define local coordinates for **all eight vertices** of a scaled block
    (corresponding to a cube with half‑extents ``(0.25, 1.0, 0.5)``) and
    transform them into world space using each block's rotation and
    translation.  A distinct colour is assigned to each corner so that
    correspondences can be inferred by colour or by feature matching.  For
    the far block, the colour indices are cyclically offset to avoid reusing
    the same ordering.

    Returns
    -------
    list of dict
        Each element has a ``position`` key (a 3‑element list) and a
        ``colour`` key (RGB triplet).  There are sixteen markers in total,
        eight on the red block and eight on the blue block.
    """
    markers: List[Dict[str, List[float]]] = []

    # Local coordinates for all eight corners of a cube scaled by (0.5, 2, 1).
    # A unit cube centred at the origin has half‑extents (1,1,1).  After
    # scaling, the half extents become (0.25, 1.0, 0.5).  We place markers
    # at all combinations (±0.25, ±1.0, ±0.5), i.e. the eight vertices of the
    # scaled block.  The order here is arbitrary but consistent across blocks.
    local_corners = [
        # Top layer (y=+1)
        (-0.25, 1.0, -0.5),  # front‑left
        (0.25, 1.0, -0.5),   # front‑right
        (0.25, 1.0, 0.5),    # back‑right
        (-0.25, 1.0, 0.5),   # back‑left
        # Bottom layer (y=-1)
        (-0.25, -1.0, -0.5), # front‑left
        (0.25, -1.0, -0.5),  # front‑right
        (0.25, -1.0, 0.5),   # back‑right
        (-0.25, -1.0, 0.5),  # back‑left
    ]

    # Colour palette for eight markers (RGB values in linear space).  We reuse
    # these colours for both blocks; to differentiate the far block, we
    # cyclically offset the index by 4 when assigning colours.  Feel free to
    # customise this palette if you need more distinct colours.
    colours = [
        [1.0, 0.0, 0.0],    # red
        [0.0, 1.0, 0.0],    # green
        [0.0, 0.0, 1.0],    # blue
        [1.0, 1.0, 0.0],    # yellow
        [1.0, 0.0, 1.0],    # magenta
        [0.0, 1.0, 1.0],    # cyan
        [0.5, 1.0, 0.0],    # chartreuse
        [0.5, 0.0, 1.0],    # violet
    ]

    # Transform for the near (red) block: rotated by 190° around Y and translated
    near_rotation = _rotation_matrix_y(math.radians(190.0))
    near_translation = np.array([-0.3, 0.0, -8.0])

    # Transform for the far (blue) block: rotated by 175° around Y and translated
    far_rotation = _rotation_matrix_y(math.radians(175.0))
    far_translation = np.array([0.3, 0.0, -9.0])

    # Generate markers for the red block (near block)
    for idx, (lx, ly, lz) in enumerate(local_corners):
        local = np.array([lx, ly, lz])
        world = near_translation + near_rotation @ local
        markers.append({
            "position": world.tolist(),
            "colour": colours[idx % len(colours)],
        })

    # Generate markers for the blue block (far block).  Offset the colour index by 4
    # to give a different colour ordering compared to the near block.
    for idx, (lx, ly, lz) in enumerate(local_corners):
        local = np.array([lx, ly, lz])
        world = far_translation + far_rotation @ local
        markers.append({
            "position": world.tolist(),
            "colour": colours[(idx + 4) % len(colours)],
        })

    return markers


def make_scene_and_pose(x_offset: float) -> Tuple[Dict, mi.ScalarTransform4f, List[float], List[float], List[float]]:
    """
    Create a Mitsuba scene dictionary for a camera translated along the x axis.

    Unlike the original implementation that panned the camera by yawing around
    the Y axis, this function keeps the camera's orientation fixed (pointing
    towards the negative Z direction) and translates it parallel to the mirror
    plane.  The translation offsets are provided by ``X_OFFSETS`` and define
    how far the camera is shifted left or right.

    Parameters
    ----------
    x_offset : float
        Horizontal translation of the camera along the world X axis (in world units).

    Returns
    -------
    scene_dict : dict
        A dictionary describing the scene, ready to be consumed by ``mi.load_dict``.
    T_wc : mi.ScalarTransform4f
        Camera‑to‑world transform for the translated camera.
    origin : list
        Camera origin in world coordinates.
    target : list
        A point on the negative Z axis that the camera's forward direction passes through.
    up : list
        Up vector defining the camera's vertical direction.
    """
    # Base orientation: camera located at (0, 0, Z0) looking at (0, 0, 0)
    T_center = mi.ScalarTransform4f.look_at(origin=[0.0, 0.0, Z0], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])

    # Apply horizontal translation in world space to the centre orientation.
    # This results in a camera that keeps the same orientation but is shifted
    # along the X axis by x_offset.
    T_wc = mi.ScalarTransform4f.translate([x_offset, 0.0, 0.0]) @ T_center

    # The origin of the translated camera in world coordinates
    origin = [x_offset, 0.0, Z0]
    # Define a target point along the negative Z axis from the camera's origin.  This
    # value is only used for metadata; it does not affect the orientation because
    # T_center already encodes the orientation.  Here we set the target so that
    # the camera points straight along the negative Z direction.
    target = [x_offset, 0.0, 0.0]
    up = [0.0, 1.0, 0.0]

    # Base scene definition without markers
    scene_dict: Dict[str, Dict] = {
        "type": "scene",
        "integrator": {"type": "path"},
        # Mirror plane at z=0
        "mirror": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate([0, 0, 0]) @ mi.ScalarTransform4f.scale([2, 2, 1]),
            "bsdf": {"type": "conductor", "material": "Ag"},
        },
        # Near red block
        "near_block": {
            "type": "cube",
            "to_world": (
                mi.ScalarTransform4f.rotate([0, 1, 0], 190)
                @ mi.ScalarTransform4f.scale([0.5, 2, 1])
                @ mi.ScalarTransform4f.translate([-0.3, 0, -8])
            ),
            # Use a textured diffuse material to aid optical flow estimation.  Instead
            # of relying on the ``scale`` texture plugin (which is not available in
            # Mitsuba 3), we pre‑tint the checkerboard texture externally and load
            # it directly here.  The red block uses ``CHECKER_TEXTURE_RED``.
            "bsdf": {
                "type": "twosided",
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "bitmap",
                        "filename": CHECKER_TEXTURE_RED,
                        "filter_type": "bilinear",
                        "wrap_mode": "repeat"
                    }
                }
            },
        },
        # Far blue block
        "far_block": {
            "type": "cube",
            "to_world": (
                mi.ScalarTransform4f.rotate([0, 1, 0], 175)
                @ mi.ScalarTransform4f.scale([0.5, 2, 1])
                @ mi.ScalarTransform4f.translate([0.3, 0, -9])
            ),
            # Similarly texture the blue block using a pre‑tinted checkerboard.
            "bsdf": {
                "type": "twosided",
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "bitmap",
                        "filename": CHECKER_TEXTURE_BLUE,
                        "filter_type": "bilinear",
                        "wrap_mode": "repeat"
                    }
                }
            },
        },
        # Soft environment light
        "emitter": {"type": "constant", "radiance": {"type": "rgb", "value": [1.2, 1.2, 1.2]}},
        # Perspective camera
        "sensor": {
            "type": "perspective",
            "to_world": T_wc,
            "fov": FOV_DEG,
            "film": {"type": "hdrfilm", "width": IMG_W, "height": IMG_H, "rfilter": {"type": "gaussian"}},
            "sampler": {"type": "independent", "sample_count": 256},
        },
    }

    # Insert fiducial markers as small coloured spheres
    marker_definitions = compute_marker_positions()
    for idx, marker in enumerate(marker_definitions):
        pos = marker["position"]
        colour = marker["colour"]
        scene_dict[f"marker_{idx:02d}"] = {
            "type": "sphere",
            # Translate to the marker position then scale the unit sphere to a larger radius.
            # A radius of 0.25 world units makes the fiducial spheres more visible in
            # the rendered image, even when viewed via the mirror.  Adjust this value
            # (e.g. 0.2–0.4) if markers appear too small or too large.
            "to_world": mi.ScalarTransform4f.translate(pos) @ mi.ScalarTransform4f.scale(0.25),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": colour},
            },
        }

    return scene_dict, T_wc, origin, target, up


def transform4f_to_numpy(T: mi.ScalarTransform4f) -> np.ndarray:
    """Convert a Mitsuba Transform4f to a NumPy 4×4 matrix."""
    return np.array(T.matrix, dtype=np.float64)


def decompose_extrinsics(T_wc_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a 4×4 world←camera transform into rotation and translation
    components.  Also return the inverse transform for projection.

    Parameters
    ----------
    T_wc_mat : np.ndarray
        4×4 transformation matrix mapping camera coordinates to world coordinates.

    Returns
    -------
    R_wc : np.ndarray (3×3)
        Rotation matrix from camera to world.
    t_wc : np.ndarray (3,)
        Translation vector from camera to world.
    R_cw : np.ndarray (3×3)
        Rotation matrix from world to camera.
    t_cw : np.ndarray (3,)
        Translation vector from world to camera.
    T_cw : np.ndarray (4×4)
        Inverse transform mapping world coordinates to camera coordinates.
    """
    R_wc = T_wc_mat[:3, :3]
    t_wc = T_wc_mat[:3, 3]
    T_cw = np.linalg.inv(T_wc_mat)
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    return R_wc, t_wc, R_cw, t_cw, T_cw


def compute_intrinsics(fov_deg: float, w: int, h: int) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Compute the camera intrinsic matrix K given the horizontal field of view
    and image size.
    """
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx  # assume square pixels
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, fx, fy, cx, cy


def ensure_dirs(path: str) -> None:
    """Ensure the output directory exists."""
    os.makedirs(os.path.join(path, "images"), exist_ok=True)


def main() -> None:
    """Render the scene for each yaw angle and save images and metadata."""
    ensure_dirs(OUT_DIR)

    # Save camera intrinsics
    K, fx, fy, cx, cy = compute_intrinsics(FOV_DEG, IMG_W, IMG_H)
    intrinsics = {
        "width": IMG_W,
        "height": IMG_H,
        "fov_deg": FOV_DEG,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": K.tolist(),
        "notes": "K assumes square pixels and horizontal FOV.",
    }
    with open(os.path.join(OUT_DIR, "intrinsics.json"), "w") as f:
        json.dump(intrinsics, f, indent=2)

    frames: List[Dict] = []
    # Iterate over horizontal translation offsets instead of yaw angles
    for idx, offset in enumerate(X_OFFSETS):
        # Build the scene and get the camera transform
        scene_dict, T_wc, origin, target, up = make_scene_and_pose(offset)
        scene = mi.load_dict(scene_dict)
        # Render the image
        img = mi.render(scene)
        # Save to disk as PNG
        frame_id = f"frame_{idx:03d}"
        png_path = os.path.join(OUT_DIR, "images", f"{frame_id}.png")
        bmp = mi.Bitmap(img).convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True,
        )
        bmp.write(png_path)
        # Extract extrinsic matrices from the sensor
        sensor = scene.sensors()[0]
        try:
            T_wc_raw = sensor.world_transform()
        except TypeError:
            # Older Mitsuba builds require an explicit time parameter
            T_wc_raw = sensor.world_transform(0.0)
        T_wc_mat = transform4f_to_numpy(T_wc_raw)
        R_wc, t_wc, R_cw, t_cw, T_cw = decompose_extrinsics(T_wc_mat)
        frames.append({
            "id": frame_id,
            "x_offset": float(offset),
            "image_path": png_path,
            "origin": origin,
            "target": target,
            "up": up,
            "T_world_from_cam": T_wc_mat.tolist(),
            "T_cam_from_world": T_cw.tolist(),
            "R_wc": R_wc.tolist(),
            "t_wc": t_wc.tolist(),
            "R_cw": R_cw.tolist(),
            "t_cw": t_cw.tolist(),
        })

    # Write extrinsics to poses.json
    poses = {
        "conventions": {
            "projection": "x ~ K [R_cw | t_cw] X_world",
            "T_world_from_cam": "4x4 transform mapping camera coordinates to world coordinates",
            "T_cam_from_world": "inverse of T_world_from_cam",
            "note": "R_cw,t_cw 用于投影；R_wc,t_wc 表示相机在世界系中的位姿。",
        },
        "frames": frames,
    }
    with open(os.path.join(OUT_DIR, "poses.json"), "w") as f:
        json.dump(poses, f, indent=2)

    print(f"[OK] Wrote {len(frames)} images, intrinsics.json and poses.json to: {OUT_DIR}")


if __name__ == "__main__":
    main()