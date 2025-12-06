# COLMAP Sparse Reconstruction & Visualization

This folder contains simple scripts to run a **sparse COLMAP SfM pipeline** on a single sequence and visualize the resulting camera trajectory and sparse point cloud.

- `capture/`: example images from 60 scenes, 1210 frames.
- `sparse_recon.sh`: runs COLMAP feature extraction, sequential matching, sparse mapping, bundle adjustment, and TXT export for one dataset.
- `visualize.py`: loads COLMAP’s `images.txt` / `points3D.txt` and plots camera centers and 3D points.

---

## 1. Requirements

- [COLMAP](https://colmap.github.io/) installed and on your `PATH`
- Python 3 with:
  - `numpy`
  - `matplotlib`

Example (conda):

```bash
pip install numpy matplotlib
```

---

## 2. Dataset Layout

Each sequence is assumed to live in a folder like:

```text
/path/to/SCENE_NAME/
    images/               # input JPEGs
    database.db           # created by COLMAP
    sparse/               # raw sparse models
    sparse_refined/       # refined model after bundle adjustment
    sparse_refined_txt/   # TXT export (cameras/images/points3D)
```

In `sparse_recon.sh`, set the `DATASET` variable at the top to your scene folder:

```bash
DATASET=/path/to/SCENE_NAME
```

---

## 3. Running Sparse COLMAP Reconstruction

The main pipeline is:

1. **Feature extraction**

   ```bash
   colmap feature_extractor \
     --database_path $DATASET/database.db \
     --image_path $DATASET/images \
     --ImageReader.single_camera 1 \
     --ImageReader.camera_model SIMPLE_RADIAL \
     --SiftExtraction.use_gpu 1
   ```

2. **Sequential matching**

   ```bash
   colmap sequential_matcher \
     --database_path $DATASET/database.db \
     --SiftMatching.use_gpu 1
   ```

3. **Incremental mapping (sparse SfM)**

   ```bash
   mkdir -p $DATASET/sparse
   colmap mapper \
     --database_path $DATASET/database.db \
     --image_path $DATASET/images \
     --output_path $DATASET/sparse \
     --Mapper.ba_refine_focal_length 0 \
     --Mapper.ba_refine_principal_point 0 \
     --Mapper.tri_min_angle 2
   ```

4. **Model stats (optional sanity check)**

   ```bash
   for i in 0 1; do
     if [ -d "$DATASET/sparse/$i" ]; then
       echo "== Model $i =="
       colmap model_analyzer --path $DATASET/sparse/$i | sed -n '1,25p'
     fi
   done
   ```

5. **Global bundle adjustment / refinement**

   ```bash
   mkdir -p $DATASET/sparse_refined
   colmap bundle_adjuster \
     --input_path  $DATASET/sparse/0 \
     --output_path $DATASET/sparse_refined \
     --BundleAdjustment.refine_focal_length 1 \
     --BundleAdjustment.refine_principal_point 0 \
     --BundleAdjustment.refine_extra_params 1
   ```

6. **TXT export**

   ```bash
   colmap model_converter \
     --input_path  $DATASET/sparse_refined \
     --output_path $DATASET/sparse_refined_txt \
     --output_type TXT
   ```

### How to run

```bash
# edit DATASET inside the script first
./sparse_recon.sh
```

After it finishes, you should see:

```
$DATASET/sparse_refined_txt/
    cameras.txt
    images.txt
    points3D.txt
```

---

## 4. Visualizing Cameras and Sparse Points

`visualize.py` provides a minimal 3D visualization of the reconstructed camera trajectory and sparse point cloud.

At the top of the script, set:

```python
TXT_DIR = "/path/to/SCENE_NAME/sparse_refined_txt"
```

The script:

* Parses `images.txt` to recover camera centers.
* Parses `points3D.txt` to load 3D points.
* Plots camera centers as a 3D line and points as a 3D scatter.
* Saves a figure as `output.png` in the current directory.

Run:

```bash
python visualize.py
```

You should get a 3D plot of the façade points and the camera path.

---

## 5. Notes

* For a new scene, create a new `SCENE_NAME/` folder with an `images/` subfolder and update `DATASET` accordingly.
* Intrinsics are assumed to come from a separate calibration step; in `bundle_adjuster` we refine focal length and extra parameters while keeping the principal point fixed.
* This pipeline only covers **sparse** reconstruction. Dense stereo (`image_undistorter`, `patch_match_stereo`, `stereo_fusion`) can be added on top if needed (but for pose estimation sparse is sufficient).
