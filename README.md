# STSRNet: Physics-Guided Super-Resolution of Reanalysis Wind Fields

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
git clone https://github.com/yourusername/ERA5WindRecon.git
cd ERA5WindRecon

# Install dependencies
pip install -r requirements.txt
```

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

```bash
python src/inference.py \
    --model_type normal \
    --checkpoint_path runs/train/normal/stsr_best.pth \
    --era5_path /path/to/era5/data/ERA5_2020-2024.nc \
    --output_path outputs/reconstructed.nc \
    --normalizer_path normalizer.json \
    --lsm_path lsm_era5.nc
```

## Model Variants

The repository includes several model variants for ablation studies:

- `normal`: Full STSRNet with all components
- `compare1`: Without multi-scale fusion
- `compare2`: Without temporal Transformer
- `compare3`: Without attention and physics gating
- `compare4`: Without LSM gating

## Results

### Performance Metrics

Validation against HadISD station observations shows:
- Mean bias reduction: 6.4%–17.4% in 4–25 m/s range
- Improved representation of coastal gradients
- Better capture of monsoon-related circulation patterns

### Visualizations

![Comparison with ERA5](images/5.compare2era5.jpg)


## Citation

If you use this code in your research, please cite:

```bibtex
@article{stsrnet2025,
  title={Physics-Guided Super-Resolution of Reanalysis Wind Fields in the Guangdong-Hong Kong-Macao Greater Bay Area},
  author={Wang, Shuxian and Liu, Xin and Wang, Yajun and Zheng, Jiajie and Ou, Suying and Wang, Jiuke and Cai, Huayang},
  journal={Nuclear Physics B},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact:
- Email: caihy7@mail.sysu.edu.cn
- GitHub Issues: [Create an issue](https://github.com/yourusername/ERA5WindRecon/issues)

## Acknowledgments

This work was supported by the Institute of Estuarine and Coastal Research, Sun Yat-Sen University, and the Southern Marine Science and Engineering Guangdong Laboratory (Zhuhai).

