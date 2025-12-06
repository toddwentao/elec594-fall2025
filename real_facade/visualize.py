# visualize_colmap.py
import numpy as np
import matplotlib.pyplot as plt

def qvec2rotmat(q):
    # q = [qw, qx, qy, qz]
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),     2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),     1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw),     1-2*(qx*qx+qy*qy)]
    ], dtype=float)

def load_images_txt(path):
    names, C_list, q_list, t_list = [], [], [], []
    with open(path, 'r') as f:
        lines = [ln for ln in f if not ln.startswith('#') and ln.strip()]
    # images.txt: Each frame occupies two lines. The first line contains the pose, and the second line contains the 2D-3D correspondence.
    for i in range(0, len(lines), 2):
        toks = lines[i].strip().split()
        # image_id qw qx qy qz tx ty tz camera_id name
        qw, qx, qy, qz = map(float, toks[1:5])
        tx, ty, tz     = map(float, toks[5:8])
        name = toks[9]
        R = qvec2rotmat(np.array([qw, qx, qy, qz], dtype=float))
        t = np.array([tx, ty, tz], dtype=float)
        C = -R.T @ t
        names.append(name); C_list.append(C); q_list.append([qw,qx,qy,qz]); t_list.append([tx,ty,tz])
    return np.array(C_list), names, np.array(q_list), np.array(t_list)

def load_points3D_txt(path, max_n=None):
    pts = []
    with open(path, 'r') as f:
        for ln in f:
            if ln.startswith('#'): continue
            toks = ln.strip().split()
            if len(toks) < 4: continue
            X, Y, Z = map(float, toks[1:4])
            pts.append([X, Y, Z])
    pts = np.array(pts) if pts else np.zeros((0,3))
    if max_n is not None and len(pts) > max_n:
        idx = np.random.choice(len(pts), max_n, replace=False)
        pts = pts[idx]
    return pts

# Edit this path to your COLMAP TXT export folder
TXT_DIR = "/workspace/data/DSC_0010/sparse_refined_txt"
C, names, q, t = load_images_txt(f"{TXT_DIR}/images.txt")
P = load_points3D_txt(f"{TXT_DIR}/points3D.txt", max_n=5000)

print(f"Registered images: {len(C)}")
print(f"First / last names: {names[:2]} ... {names[-2:]}")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# Camera trajectory
ax.plot(C[:,0], C[:,1], C[:,2], marker='o')
# Sparse point cloud
if len(P):
    ax.scatter(P[:,0], P[:,1], P[:,2], s=1)

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('COLMAP cameras & sparse points')

plt.savefig('/workspace/output.png', bbox_inches='tight')

