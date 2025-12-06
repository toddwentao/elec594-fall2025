import mitsuba as mi
import numpy as np
import json
import math

# Resolution & FOV
W, H = 1280, 720
fov = 90 # degrees

fx = (W / 2) / math.tan(math.radians(fov / 2))
fy = fx * (W / H)
cx = W / 2
cy = H / 2

# Camera motion parameters (same as your script)
num_frames = 90
step = 5
x = -1.0
y = 1.5
step_z = 0.03
radius = 0.5

output_path = "/home/elec594/Desktop/luigi/Render/classroom/transforms.json"

def look_at_matrix(origin, target, up=[0,1,0]):
    origin = np.array(origin)
    target = np.array(target)
    up = np.array(up)

    forward = target - origin
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    new_up = np.cross(right, forward)

    # Camera-to-world
    R = np.stack([right, new_up, -forward], axis=1)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = origin
    return M

frames = []

for i in range(0, num_frames, step):
    z = -1.5 + step_z * i
    theta = (i / num_frames) * 0.5*np.pi + 0.25*np.pi
    
    origin = [x, y, z]
    target = [x + math.sin(theta)*radius, y, z + math.cos(theta)*radius]

    M = look_at_matrix(origin, target)

    frames.append({
        "file_path": f"frame_{i:03d}.png",
        "transform_matrix": M.tolist()
    })

data = {
    "camera_model": "OPENCV",
    "fl_x": fx,
    "fl_y": fy,
    "cx": cx,
    "cy": cy,
    "w": W,
    "h": H,
    "frames": frames
}

with open(output_path, "w") as f:
    json.dump(data, f, indent=2)


print("transforms.json written!")