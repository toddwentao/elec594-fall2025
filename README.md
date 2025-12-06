# Learning Reflections: Glass Facades for Urban-Scale (Rice ELEC 594)

This repository contains the codebase and resources for the Rice University ELEC 594 final project: **"Learning Reflections: Glass Facades for Urban-Scale"**.

## Project Overview

Glass buildings and mirror-like surfaces in urban environments create complex reflections that challenge existing 3D reconstruction systems (e.g., COLMAP, NeRF). This project aims to develop a reflection-aware 3D reconstruction framework that transforms reflections from obstacles into valuable sensing information. By combining geometric surface recovery of glass facades with machine learning–based reflection modeling, our approach enables simultaneous reconstruction of true building geometry and extraction of environmental information from reflected imagery.

**Key Components:**
1.  **GEM (Glass-Segmentor):** A segmentation module to robustly identify glass regions in images.
2.  **Simulation Pipeline:** A Mitsuba-based renderer for generating physically realistic reflective scenes (flat and distorted mirrors).
3.  **Mirror Correction:** A pipeline to correct "phantom" geometries caused by reflections using VGGT predictions and geometric constraints.
4.  **Real-World Dataset:** A collection of reflection sequences from urban facades for validation.

## Team Members

*   **Shin Li** (zl186@rice.edu)
*   **Chi Xu** (cx31@rice.edu)
*   **Yunan Wang** (yw256@rice.edu)
*   **Yixing Liu** (yl366@rice.edu)
*   **Wentao Jiang** (wj30@rice.edu)

## Directory Structure

*   `GEM/`: Code for the Glass Segmentation Module (based on MaskDINO/Detectron2).
*   `3d_recon/`: Python scripts for 3D reconstruction and synthetic data generation using Mitsuba 3.
    *   `single_object.py`: Renders synthetic scenes with reflective objects.
    *   `create_distort_glass.py`: Generates distorted glass geometries.
*   `real_facade/`: Tools and scripts for processing real-world capture data.
    *   `capture/`: Contains captured image sequences (DSC_xxxx).
    *   `sparse_recon.sh`: Shell script for running COLMAP reconstruction.
*   `ELEC_594_Final_Report/`: LaTeX source code and figures for the final project report.

## Setup & Usage

### 1. Glass Segmentation (GEM)

The `GEM` directory contains the segmentation framework. Refer to `GEM/README.md` for detailed installation and usage instructions.

```bash
# Example usage for GEM
cd GEM
# Install dependencies (see GEM/README.md)
python demo/demo.py --config-file configs/gsd-s/semantic-segmentation/gem_sam_base_bs16_iter2w_steplr.yaml ...
```

### 2. Synthetic Data Generation (Mitsuba 3)

The `3d_recon` folder contains scripts to generate synthetic datasets with controlled reflection properties.

**Requirements:**
*   Mitsuba 3
*   NumPy
*   FFmpeg (for video generation)

**Running the Simulation:**

```bash
# Generate a distorted glass scene render
python 3d_recon/single_object.py
```

This script renders a sequence of frames simulating a camera moving around a reflective object (flat or distorted plane) and saves the output to `3d_recon/output/`.

### 3. Real-World Reconstruction (COLMAP)

The `real_facade` directory provides tools to process real-world images using COLMAP.

**Requirements:**
*   COLMAP installed and accessible in your PATH.

**Running Reconstruction:**

```bash
# Edit DATASET path in the script first
bash real_facade/sparse_recon.sh
```

## Acknowledgements

We thank **Professor Vivek Boominathan** for his guidance. We also acknowledge the open-source communities behind GEM, COLMAP, and Mitsuba.

