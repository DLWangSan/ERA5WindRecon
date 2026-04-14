# Project Structure

```
ERA5WindRecon/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── dataset.py         # Dataset class for ERA5 wind fields
│   ├── models.py          # STSRNet model architectures
│   ├── train.py           # Training script
│   └── inference.py       # Inference script
├── images/                # Figures and visualizations
│   ├── 0.abstract.png
│   ├── 1.area.png
│   ├── 2.nn.jpg
│   ├── 3.ablation_four_panel.png
│   ├── 4.correlations.png
│   ├── 5.compare2era5.jpg
│   ├── 6.compare_season_flower.png
│   ├── 7.divergence.png
│   ├── 8.wind_field_tool_ui.jpg
│   └── 9.compare2hadisd.png
├── data/                  # Data directory (not included in repo)
├── README.md              # Main documentation
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
├── .gitignore            # Git ignore rules
├── example_usage.py      # Example usage script
└── STRUCTURE.md          # This file

```

## Key Files

- **src/dataset.py**: Implements `ERA5WindSRDataset` for loading and preprocessing ERA5 data
- **src/models.py**: STSRNetPlus and related building blocks
- **src/train.py**: Training script with physics-guided loss functions
- **src/inference.py**: Script for generating reconstructed NetCDF files






