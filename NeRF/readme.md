# 3D Mirror Surface Reconstruction – Simulation & NeRF Training Pipeline

This repository contains a complete pipeline for simulating reflective surfaces, generating distorted-glass geometry, reconstructing surface normals, and training NeRF models on physically simulated mirror reflection datasets. All simulations use Mitsuba 3, and NeRF training is performed using Nerfstudio.

---

## Project Structure Overview

file/
│
├── nerf-pytorch/                     # Official NeRF (yenchenlin/nerf-pytorch)
│
├── datasets/
│   ├── distorted_lens_frames/        # Simulated distorted-mirror images for NeRF training
│   └── flat_lens_frames/             # Simulated flat-mirror images for NeRF training
│
├── Output_Colab_distorted_90/        # Distorted-mirror NeRF results (90-frame training)
│   ├── point_cloud.ply
│   ├── renders/
│   └── nerfstudio_models/
│
├── Output_Colab_flat_90/             # Flat-mirror NeRF results (90-frame training)
│   ├── point_cloud.ply
│   ├── renders/
│   └── nerfstudio_models/
│
├── export_pc/                        # Scripts for exporting NeRF point clouds
├── depth_output/                     # Additional NeRF depth rendering
├── renders/                          # Rendering utilities
│
├── config-distorted.yml              # Nerfstudio config for distorted mirror
├── config-flat.yml                   # Nerfstudio config for flat mirror
│
├── NeRF.py                           # Utilities for Nerfstudio API
└── render_depth.py                   # Script to render NeRF depth maps

---

## Dataset Description

### Distorted Mirror Dataset
Path:
`/Users/young/Documents/matata/code/NeRF/datasets/distorted_lens_frames`

This dataset contains reflection images rendered from a spatially varying distorted mirror surface.

### Flat Mirror Dataset
Path:
`/Users/young/Documents/matata/code/NeRF/datasets/flat_lens_frames`

This dataset contains reflection images rendered from an ideal flat mirror.

---

# How to Train NeRF (Google Colab, A100, Nerfstudio)

NeRF training in this project is performed using Nerfstudio on Google Colab with an NVIDIA A100 GPU.

## 1. Environment Setup

Enable GPU: # 3D Mirror Surface Reconstruction – Simulation & NeRF Training Pipeline

This repository contains a complete pipeline for simulating reflective surfaces, generating distorted-glass geometry, reconstructing surface normals, and training NeRF models on physically simulated mirror reflection datasets. All simulations use Mitsuba 3, and NeRF training is performed using Nerfstudio.


Enable GPU: Runtime → Change runtime type → GPU (A100)

Install Nerfstudio:
```bash
pip install nerfstudio
ns-install-cli

from google.colab import drive
drive.mount('/content/drive')

2. Prepare Dataset Paths

/content/flat_lens_frames/distorted_lens_frames
/content/flat_lens_frames/flat_lens_frames

3.Train NeRF

ns-train nerfacto \
    --vis viewer_legacy \
    --max-num-iterations 30000 \
    --pipeline.model.predict_normals True \
    --output-dir "/content/drive/MyDrive/Colab Notebooks/flat_lens_frames/nerf_normals_output_distorted" \
    nerfstudio-data \
    --data "/content/flat_lens_frames/distorted_lens_frames"

4. Export NeRF Point Cloud

ns-export pointcloud \
    --load-config "/content/drive/MyDrive/Colab Notebooks/flat_lens_frames/nerf_normals_output_distorted/nerfstudio_models/config.yml" \
    --output-dir "/content/drive/MyDrive/Colab Notebooks/flat_lens_frames/output_pc" \
    --num-points 300000