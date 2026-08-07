# 3D Object Extraction and Rendering Pipeline

This repository contains a three-step Python pipeline designed to work with 3D Gaussian Splatting (or point cloud) representations of a scene. The pipeline allows you to isolate specific objects from a full 3D scene using 2D masks, generate a smooth camera trajectory for novel views, and finally render RGB images and segmentation masks of the isolated objects from those novel views.

## Pipeline Overview

1. **Extract Objects** (`01_extract_objects.py`): Isolates specific objects from a scene `.ply` file based on multi-view 2D masks.
2. **Generate Camera Trajectory** (`02_generate_camera_trajectory.py`): Creates a new `cameras.json` by interpolating between original camera poses to create a smooth path.
3. **Render and Segment** (`03_render_and_segment.py`): Merges the isolated objects, assigns them segmentation labels, and renders RGB and segmentation masks from the novel camera trajectory.

---

## Prerequisites

Ensure you have the following Python dependencies installed:
- `numpy`
- `torch`
- `Pillow` (PIL)
- `plyfile`

---

## 1. Extract Objects (`01_extract_objects.py`)

This script projects the 3D points/Gaussians into 2D camera views and tallies "votes" to determine which points consistently fall within the provided 2D masks. It outputs a separate `.ply` file for each object.

### Usage
```bash
python 01_extract_objects.py \
    --ply scene_output.ply \
    --cameras cameras.json \
    --masks masks-ball masks-cupboard masks-cupboard-2 \
    --out extracted_objects
```

### Arguments
- `--ply`: Path to the input 3D scene PLY file (default: `scene_output.ply`)
- `--cameras`: Path to the original `cameras.json` (default: `cameras.json`)
- `--masks`: One or more directories containing 2D mask images corresponding to the camera views. The script expects the mask images to have the same base names as the images in `cameras.json` (default: `["masks-ball", "masks-cupboard", "masks-cupboard-2"]`)
- `--out`: Directory to save the extracted object `.ply` files (default: `extracted_objects`)
- `--gamma`: Threshold for the voting mechanism (default: `0.1`)
- `--views`: Number of views to sample for voting. If `-1`, uses all views (default: `-1`)

---

## 2. Generate Camera Trajectory (`02_generate_camera_trajectory.py`)

This script takes the original discrete camera poses and creates a smooth, continuous trajectory by interpolating between them (using SLERP for rotations). 

### Usage
```bash
python 02_generate_camera_trajectory.py \
    --cameras cameras.json \
    --outfile novel_cameras.json \
    --n_views 36
```

### Arguments
- `--cameras`: Path to the original `cameras.json` (Required)
- `--outfile`: Path to save the interpolated camera path (default: `novel_cameras.json`)
- `--n_views`: The number of novel views to generate in the smooth trajectory (default: `36`)

---

## 3. Render and Segment (`03_render_and_segment.py`)

This final script takes the individual `.ply` files extracted in Step 1, merges them with distinct label IDs, and uses a PyTorch-based rasterizer to render RGB images and 2D segmentation masks from the novel camera views generated in Step 2.

### Usage
```bash
python 03_render_and_segment.py \
    --ball extracted_objects/masks-ball.ply \
    --cup1 extracted_objects/masks-cupboard.ply \
    --cup2 extracted_objects/masks-cupboard-2.ply \
    --cameras novel_cameras.json \
    --outdir output
```

### Arguments
- `--ball`: Path to the extracted ball `.ply` (Required)
- `--cup1`: Path to the extracted first cupboard `.ply` (Required)
- `--cup2`: Path to the extracted second cupboard `.ply` (Required)
- `--cameras`: Path to the `novel_cameras.json` generated in step 2 (Required)
- `--outdir`: Directory to save the rendered `rgb/` and `masks/` images (default: `output`)
- `--n_views`: Number of views expected in the cameras file (default: `36`)
- `--bg`: RGB values for the background color (default: `0. 0. 0.`)

### Output Format
Inside the `--outdir` (e.g., `output`), two subdirectories will be created:
- `rgb/`: Contains the rendered RGB images (`view_XXXX_rgb.png`).
- `masks/`: Contains binary masks for each object and a color-coded overlay (`view_XXXX_mask_<object>.png`, `view_XXXX_mask_overlay.png`).
