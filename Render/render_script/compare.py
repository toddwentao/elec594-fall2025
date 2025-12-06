import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')


def depth_metrics(pred, gt, mask=None):
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    valid = gt > 0
    pred = pred[valid]
    gt = gt[valid]

    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    abs_rel = np.mean(np.abs(pred - gt) / gt)

    ratio = np.maximum(pred / gt, gt / pred)
    delta1 = np.mean(ratio < 1.25)

    return {'RMSE': rmse, 'AbsRel': abs_rel, 'delta1': delta1}

def depth_metrics_normalized(pred, gt, mask=None, delta_thresh=0.05):
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    valid = gt > 0
    pred = pred[valid]
    gt = gt[valid]

    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    mae = np.mean(np.abs(pred - gt))
    delta1 = np.mean(np.abs(pred - gt) < delta_thresh)

    return {'RMSE': rmse, 'MAE': mae, f'delta<{delta_thresh}': delta1}


def normalize_depth(depth_map):
    """Normalize depth map to [0, 1] range for visualization."""
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    normalized = (depth_map - min_depth) / (max_depth - min_depth + 1e-8)
    return normalized

def compare_depth_maps(depth_map1, depth_map2, title1='Depth Map 1', title2='Depth Map 2'):
    """Compare two depth maps side by side."""
    H, W = depth_map1.shape
    depth_map1 = np.clip(depth_map1, 0, 12)
    

    norm_depth1 = normalize_depth(depth_map1)
    norm_depth2 = normalize_depth(depth_map2)

    # Same shape check
    if depth_map1.shape != depth_map2.shape:
        print(f"Warning: Depth maps have different shapes: {depth_map1.shape} vs {depth_map2.shape}")
        # Resize depth_map2 to match depth_map1
        from skimage.transform import resize
        norm_depth2 = resize(norm_depth2, (H, W), preserve_range=True)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(norm_depth1, cmap='plasma')
    plt.colorbar()
    plt.title(title1)

    plt.subplot(1, 2, 2)
    plt.imshow(norm_depth2, cmap='plasma')
    plt.colorbar()
    plt.title(title2)

    plt.tight_layout()
    plt.show()
    plt.savefig('depth_map_comparison.png')

    print("Depth Map 1 Metrics:", depth_metrics_normalized(norm_depth2, norm_depth1))

depth_map_1 = np.load('/home/elec594/Desktop/luigi/Render/classroom/lens_frames/depth/frame_040_depth.npy')

# plot depth_map1 distribution
plt.figure()
plt.hist(depth_map_1.flatten(), bins=100, range=(0, 60))
plt.title('Depth Map 1 Distribution')
plt.xlabel('Depth Value')
plt.ylabel('Frequency')
plt.savefig('depth_map_1_distribution.png')

import json

depth_map_2 = json.load(open('/home/elec594/Desktop/luigi/vggt/results/classroom/predictions.json'))
depth_map_2 = depth_map_2['depth'][0][8]
print(type(depth_map_2))
depth_map_2 = np.array(depth_map_2)
print(depth_map_1.shape, depth_map_2.shape)
compare_depth_maps(depth_map_1.squeeze(), depth_map_2.squeeze(),
                   title1='Classroom Depth Map - Frame 040',
                   title2='Classroom Depth Predict - VGGT - Frame 040')