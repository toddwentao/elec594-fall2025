"""
pixel_match.py - Theoretical homography computation
===================================================

This script loads camera intrinsics and extrinsics from
``mirror_scene/intrinsics.json`` and ``mirror_scene/poses.json`` and
computes the theoretical homography matrices between adjacent frames
for a planar scene at ``Z = 0``.  The homography between two views
encodes how points on a plane in the world map from one image to
another under perspective projection.

The computation implemented here follows the standard derivation for
a single plane.  If the camera intrinsic matrix is ``K`` and the
extrinsic parameters for view ``i`` are given by rotation ``R_cw`` and
translation ``t_cw`` (mapping world coordinates to camera
coordinates), then for a plane located at ``Z=0`` we form the matrix

``G_i = [R_cw[:, :2], t_cw]``,

where ``R_cw[:, :2]`` extracts the first two columns of the rotation
matrix and ``t_cw`` is the translation vector.  The homography from
view ``i`` to view ``j`` is then

``H_ij = K @ G_j @ inv(G_i) @ inv(K)``,

and is normalized so that ``H_ij[2, 2] == 1``.  For the three-frame
sequence used in this project, the code computes ``H01`` for frame
0→1 and ``H12`` for frame 1→2.  To run the script, simply execute

    python pixel_match.py

from the ``Optical_Flow`` directory.  It will print both
homographies to the console.

"""

import json
import os
import numpy as np


def compute_theoretical_homographies(intrinsics_path: str, poses_path: str):
    """Compute theoretical homographies between adjacent frames.

    Args:
        intrinsics_path: Path to a JSON file containing the intrinsic
            matrix under the key ``"K"``.
        poses_path: Path to a JSON file containing the camera
            extrinsics for each frame under ``frames``.  Each frame
            entry must have ``R_cw`` (3×3 rotation) and ``t_cw`` (3×1
            translation) arrays.

    Returns:
        A tuple of two 3×3 numpy arrays ``(H01, H12)`` representing
        the homography from frame 0→1 and from frame 1→2 respectively.
    """
    # Load intrinsics
    with open(intrinsics_path, 'r') as f:
        intr = json.load(f)
    K = np.array(intr['K'], dtype=float)

    # Load poses
    with open(poses_path, 'r') as f:
        poses = json.load(f)
    frames = poses['frames']

    # Build G matrices for each frame: G = [R_cw[:,0:2], t_cw]
    Gs = []
    for frame in frames:
        R_cw = np.array(frame['R_cw'], dtype=float)
        t_cw = np.array(frame['t_cw'], dtype=float).reshape(3, 1)
        # concatenate first two columns of R and translation
        G = np.hstack((R_cw[:, :2], t_cw))
        Gs.append(G)

    def _compute_h(G_i: np.ndarray, G_j: np.ndarray) -> np.ndarray:
        """Compute homography from view i to view j."""
        H = K @ G_j @ np.linalg.inv(G_i) @ np.linalg.inv(K)
        # Normalize so that bottom-right element is 1
        H /= H[2, 2]
        return H

    # Compute homographies between consecutive frames
    H01 = _compute_h(Gs[0], Gs[1])
    H12 = _compute_h(Gs[1], Gs[2])

    return H01, H12


def main():
    """Run the homography computation and display results."""
    # Determine the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mirror_dir = os.path.join(script_dir, 'mirror_scene')

    # Paths to intrinsics and pose files relative to this script
    intrinsics_path = os.path.join(mirror_dir, 'intrinsics.json')
    poses_path = os.path.join(mirror_dir, 'poses.json')

    # Compute homographies
    H01, H12 = compute_theoretical_homographies(intrinsics_path, poses_path)

    # Display the results with clear labelling
    np.set_printoptions(precision=8, suppress=True)
    print("Theoretical homography H01 (frame 0 → frame 1):")
    print(H01)
    print("\nTheoretical homography H12 (frame 1 → frame 2):")
    print(H12)


if __name__ == '__main__':
    main()