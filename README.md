# STSRNet: A physics-guided deep learning framework for spatiotemporal super-resolution of coastal ERA5 wind fields

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A physics-guided spatiotemporal super-resolution network (STSRNet) for enhancing the resolution and physical consistency of ERA5 near-surface winds over coastal regions.

## Overview

STSRNet enhances both temporal and spatial resolution of ERA5 wind fields:
- **Temporal resolution**: from 1 hour to 10 minutes
- **Spatial resolution**: from 0.25° × 0.25° to approximately 0.1° × 0.1°

The model incorporates physics-guided constraints including:
- Divergence-based regularization for approximate mass conservation
- Land-sea mask gating mechanism for realistic coastal transitions

## Key Features

- **Physics-Informed Learning**: Embeds physical constraints (mass conservation, land-sea boundaries) directly into the loss function
- **Multi-Scale Architecture**: Combines 3D-CNN encoder, dilated convolutions, and temporal Transformer
- **Coastal Optimization**: Specifically designed for complex coastal wind fields with strong gradients
- **Validated Performance**: 6.4%–17.4% reduction in mean bias compared to ERA5 in the 4–25 m/s wind speed range

## Architecture

The STSRNet framework consists of three main components:

1. **Data-driven Backbone**: 3D-CNN encoder with multi-scale fusion and temporal Transformer
2. **Physics-Guided Mechanism**: Land-sea mask gating (phyGATE) and divergence regularization
3. **Spatiotemporal Upsampling**: Temporal and spatial resolution enhancement

![Network Architecture](images/2.nn.jpg)

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/DLWangSan/ERA5WindRecon.git
cd ERA5WindRecon

# (Optional) create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Install a **PyTorch** build that matches your machine ([pytorch.org](https://pytorch.org/get-started/locally/)) if `pip install -r requirements.txt` does not already give you CUDA. A GPU is recommended; CPU runs but is slower. The list includes `netcdf4` so `xarray` can read `.nc` files.

## Quick start (inference)

The repository includes a small **example** ERA5 slice and a land–sea mask under [`example_data/`](example_data/) so you can run inference without preparing full regional archives first.

1. **Environment.** From the repo root, use `pip` as in [Installation](#installation). No Conda is required.

2. **Bundled checkpoint.** This repo includes a trained weight file and matching statistics so you can run inference without training first:
   - `runs/train/normal/stsr_best.pth`
   - `normalizer.json` (must pair with that checkpoint)

3. **Run inference** on the example NetCDF and LSM. In `src/inference.py`, **`--t_in` defaults to 18** and **`--t_out` defaults to 36** (same as training); you only need to pass them if you use other values.

   ```bash
   python src/inference.py \
       --model_type normal \
       --checkpoint_path runs/train/normal/stsr_best.pth \
       --normalizer_path normalizer.json \
       --era5_path example_data/ERA5_2026_04.nc \
       --lsm_path example_data/lsm_era5.nc \
       --output_path outputs/reconstructed.nc \
       --plot_path outputs/wind_snapshot_t0.png
   ```

   **Windows (PowerShell / CMD), one line — copy as a single command:**

   ```text
   python src/inference.py --model_type normal --checkpoint_path runs/train/normal/stsr_best.pth --normalizer_path normalizer.json --era5_path example_data/ERA5_2026_04.nc --lsm_path example_data/lsm_era5.nc --output_path outputs/reconstructed.nc --plot_path outputs/wind_snapshot_t0.png
   ```

   Run from the repository root. Do **not** use `--denormalize_uv` unless you know the network outputs normalized wind (default training uses physical m/s targets).

   To rebuild `example_data/lsm_era5.nc` on the same grid as the example ERA5 file, run: `python scripts/write_example_lsm.py`.

   To train your own model instead, see [Training](#training).

## Data Preparation

### Input Data

The model requires ERA5 NetCDF files with the following variables:
- `u10`: 10-meter zonal wind component
- `v10`: 10-meter meridional wind component
- `msl`: Mean sea level pressure
- `t2m`: 2-meter air temperature

### Land-Sea Mask

An optional land-sea mask (LSM) file can be provided to improve coastal transitions.

## Usage

### Training

```bash
python src/train.py \
    --era5_path /path/to/era5/data \
    --filename ERA5_2020-2024.nc \
    --model_type normal \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --lsm_path lsm_era5.nc
```

### Inference

`--t_in` and `--t_out` default to **18** and **36** in `src/inference.py`; override only if your checkpoint was trained with different windows.

```bash
python src/inference.py \
    --model_type normal \
    --checkpoint_path runs/train/normal/stsr_best.pth \
    --era5_path /path/to/era5/data/ERA5_2020-2024.nc \
    --output_path outputs/reconstructed.nc \
    --normalizer_path normalizer.json \
    --lsm_path lsm_era5.nc
```

## Results

### Performance Metrics

Validation against HadISD station observations shows:
- Mean bias reduction: 6.4%–17.4% in 4–25 m/s range
- Improved representation of coastal gradients
- Better capture of monsoon-related circulation patterns

### Visualizations

![Comparison with ERA5](images/5.compare2era5.jpg)





## Contact

For questions or issues, please contact:
- Email: wangshx59@mail2.sysu.edu.cn
- GitHub Issues: [Create an issue](https://github.com/DLWangSan/ERA5WindRecon/issues)








