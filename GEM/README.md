# Use GEM to Segment Glass 
## Author: Chi Xu

**Note:** This project is based on the [GEM](https://github.com/isbrycee/GEM) repository.

GEM (Glass-Segmentor) is a simple but accurate segmentation framework designed specifically for glass surface segmentation. It leverages Vision Foundation Models (like SAM) and a large-scale synthesized glass surface dataset (S-GSD) to achieve state-of-the-art performance.

<img src="figures/framework.jpg" width="900px">

## Features

* **Unified Architecture**: A simple network boosting glass surface segmentation.
* **S-GSD Dataset**: Automatically constructed large-scale synthesized glass surface dataset with precise mask annotations.
* **SOTA Performance**: Surpasses previous methods by a large margin (IoU +2.1%).

## Installation

The code is based on [MaskDINO](https://github.com/IDEA-Research/MaskDINO) and [detectron2](https://github.com/facebookresearch/detectron2).

   Ensure you have PyTorch and CUDA installed. Then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: You may need to install `detectron2` manually if not included in requirements.*

## Usage

### 1. Inference / Demo

To run inference on a video or image using a pre-trained model:

**Video Inference:**
```bash
python demo/demo.py --config-file configs/gsd-s/semantic-segmentation/gem_sam_base_bs16_iter2w_steplr.yaml \
  --video-input /path/to/video.MOV \
  --output results/ \
  --opts MODEL.WEIGHTS weight/GEM-Base-S-GSD-20x-IoU.774
```

**Image Inference:**
```bash
python demo/demo.py --config-file configs/gsd-s/semantic-segmentation/gem_sam_base_bs16_iter2w_steplr.yaml \
  --input /path/to/image.jpg \
  --output results/ \
  --opts MODEL.WEIGHTS weight/GEM-Base-S-GSD-20x-IoU.774
```

## Acknowledgement

* [GEM](https://github.com/isbrycee/GEM) - This project builds upon the official GEM implementation.
* [Mask DINO](https://github.com/IDEA-Research/MaskDINO)
* [Segment Anything](https://github.com/facebookresearch/segment-anything)

