import torch
# Download the video
video_path = '/home/elec594/Desktop/luigi/Render/kitchen/output.mp4'
frame_path = '/home/elec594/Desktop/luigi/Render/kitchen/frames'
import imageio.v3 as iio
# frames = iio.imread(video_path, plugin="FFMPEG")  # plugin="pyav"
frames = []
for i in range(0, 170, 3):
    frame = iio.imread(f"{frame_path}/frame_{i:03d}.png")
    frames.append(frame)

device = 'cuda'
grid_size = 20
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

# Run Offline CoTracker:
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1


import numpy as np
np.save('/home/elec594/Desktop/luigi/Render/kitchen/tracks.npy', pred_tracks.cpu().numpy())
np.save('/home/elec594/Desktop/luigi/Render/kitchen/visibility.npy', pred_visibility.cpu().numpy())


from cotracker.utils.visualizer import Visualizer

vis = Visualizer(save_dir="/home/elec594/Desktop/luigi/Render/kitchen/saved_videos", pad_value=120, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility)