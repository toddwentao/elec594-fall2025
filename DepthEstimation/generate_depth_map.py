import numpy as np
import cv2

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def cluster_trajectories(lengths, k_max=10, plot_elbow=True):
    """
    Cluster trajectories by their total movement length using KMeans.
    Uses the Elbow method to find an optimal k.

    Args:
        lengths: np.ndarray of shape [N], each element is trajectory length
        k_max: maximum number of clusters to test
        plot_elbow: whether to plot the inertia vs k curve

    Returns:
        labels: np.ndarray of shape [N], cluster id for each trajectory
        best_k: int, chosen number of clusters
    """
    lengths = lengths.reshape(-1, 1)
    inertias = []

    # --- Run KMeans for k = 1..k_max ---
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(lengths)
        inertias.append(km.inertia_)

    # --- Determine elbow (simple curvature heuristic) ---
    # Compute 2nd derivative to find elbow
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    best_k = np.argmin(diffs2) + 2  # add offset for diff indexing

    if plot_elbow:
        plt.figure()
        plt.plot(range(1, k_max + 1), inertias, 'o-')
        plt.axvline(best_k, color='r', linestyle='--')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title(f'Elbow method (best k={best_k})')
        plt.savefig('KMeans.png')
        plt.show()

    # best_k = 2
    # --- Run final clustering ---
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    raw_labels = kmeans.fit_predict(lengths)

    # --- Step 4: Reorder clusters by mean motion length (larger motion → nearer) ---
    means = [np.mean(lengths[raw_labels == i]) for i in range(best_k)]
    sorted_clusters = np.argsort(means)[::-1]  # descending: largest motion = nearest

    # mapping: cluster_id → depth_order
    reorder_map = {old: new for new, old in enumerate(sorted_clusters)}
    depth_labels = np.array([reorder_map[label] for label in raw_labels])

    return depth_labels, best_k

def generate_coarse_depth_map(img, coords, foreground_indices, labels, trajectory_lengths=None, background_thresh=250):
    """
    Generate a coarse depth map using parallax assumption:
    objects that move more (larger trajectory length) are nearer.

    Args:
        img: np.ndarray, shape (H, W, 3)
        coords: np.ndarray, shape (N, 2), (x, y) coordinates of all tracked points
        foreground_indices: np.ndarray of indices indicating foreground points
        labels: np.ndarray, shape (len(foreground_indices),), cluster label of each foreground trajectory
        trajectory_lengths: np.ndarray, shape (N,), total motion length for each trajectory (optional)
        background_thresh: intensity threshold to detect background (white)

    Returns:
        depth_map: np.ndarray, shape (H, W), discrete depth levels (1..K)
    """

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    foreground_mask = gray < background_thresh  # True = non-background

    # --- Foreground coordinates and labels ---
    fg_coords = coords[foreground_indices]  # shape: (M, 2)
    fg_labels = labels                      # shape: (M,)

    sorted_labels = sorted(np.unique(fg_labels))

    label_to_depth = {lab: i + 1 for i, lab in enumerate(sorted_labels)}  # 1 = nearest

    # --- Create KDTree for nearest-neighbor assignment ---
    tree = cKDTree(fg_coords)

    # Get all foreground pixel coordinates
    ys, xs = np.nonzero(foreground_mask)
    pix_coords = np.stack([xs, ys], axis=1)

    # Query nearest foreground trajectory for each pixel
    dists, nn_idx = tree.query(pix_coords, k=1)
    nearest_labels = fg_labels[nn_idx]
    nearest_depths = np.array([label_to_depth[l] for l in nearest_labels], dtype=np.uint8)

    # --- Create depth map ---
    depth_map = np.zeros((H, W), dtype=np.uint8)
    depth_map[ys, xs] = nearest_depths

    return depth_map

tracks = np.load('/home/elec594/Desktop/luigi/Render/bedroom/tracks.npy')  

tracks = tracks[0]  

frame_idx = 27
coords = tracks[frame_idx]  # shape: [400, 2]
coords = np.round(coords).astype(int)
print(coords.shape)

img = cv2.imread('/home/elec594/Desktop/luigi/Render/bedroom/frames/frame_027.png', cv2.IMREAD_COLOR)  # shape: [H, W, 3]
if img is None:
    raise FileNotFoundError("Could not find frame_027.png")

# --- Mask: remove points that fall outside the image ---
h, w = img.shape[:2]
# valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < w) & \
#              (coords[:, 1] >= 0) & (coords[:, 1] < h)

# coords = coords[valid_mask]
# valid_indices = np.where(valid_mask)[0]
# print(valid_indices.shape)

# # --- Check if pixel is white (background) ---
# pixel_colors = img[coords[:, 1], coords[:, 0]]  # note y, x order
# is_white = np.all(pixel_colors >= 250, axis=1)  # allow small tolerance

# # --- Keep only non-background points ---
# foreground_indices = valid_indices[~is_white]
# print(foreground_indices.shape)

foreground

# --- Compute moving vector for the l ---
diffs = np.diff(tracks[:, foreground_indices, :], axis=0)
distances = np.linalg.norm(diffs, axis=2)
trajectory_lengths = np.sum(distances, axis=0)

plt.figure(figsize=(7, 4))
plt.hist(trajectory_lengths, bins=30, color='skyblue', edgecolor='k', alpha=0.8)
plt.xlabel('Trajectory Length (pixels)')
plt.ylabel('Count')
plt.title('Distribution of Trajectory Lengths')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('distribution.png')
plt.show()


print(f"Kept {len(foreground_indices)} foreground trajectories out of 400")


lables, best_k = cluster_trajectories(trajectory_lengths)
print(len(lables))

depth_map = generate_coarse_depth_map(img, coords, foreground_indices, lables,)

plt.figure()
plt.imshow(depth_map, cmap='plasma')
plt.title(f"Coarse Depth Map (discrete levels)")
plt.colorbar(label="Depth Group")
plt.savefig('depth_map_9.png')
plt.show()



