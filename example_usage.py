"""
Example usage of STSRNet for wind field super-resolution.

This script demonstrates how to use the trained model for inference.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from inference import load_model, load_normalizer, generate_reconstructed_nc
from dataset import ERA5WindSRDataset
import torch

# Example configuration
CONFIG = {
    'model_type': 'normal',
    'checkpoint_path': 'runs/train/normal/stsr_best.pth',
    'normalizer_path': 'normalizer.json',
    'era5_path': 'data/ERA5_2020-2024.nc',
    'lsm_path': 'data/lsm_era5.nc',
    'output_path': 'outputs/reconstructed.nc',
    'batch_size': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def main():
    print("STSRNet Inference Example")
    print("=" * 50)
    
    # Load dataset
    print("Loading dataset...")
    dataset = ERA5WindSRDataset(
        nc_path=CONFIG['era5_path'],
        t_in=18,
        t_out=36,
        t_scale=2,
        s_scale=2,
        use_coord=True,
        lsm_path=CONFIG['lsm_path']
    )
    
    # Load model
    print("Loading model...")
    model = load_model(
        model_type=CONFIG['model_type'],
        checkpoint_path=CONFIG['checkpoint_path'],
        t_scale=12,
        s_scale=2,
        extra_scale=2.5,
        c_in=7,
        use_coord=True,
        use_lsm=True,
        device=CONFIG['device']
    )
    
    # Load normalizer
    print("Loading normalizer...")
    normalizer = load_normalizer(CONFIG['normalizer_path'], device=CONFIG['device'])
    
    # Generate reconstructed NetCDF
    print("Generating reconstructed wind fields...")
    generate_reconstructed_nc(
        model=model,
        dataset=dataset,
        normalizer=normalizer,
        output_path=CONFIG['output_path'],
        batch_size=CONFIG['batch_size'],
        device=CONFIG['device']
    )
    
    print("Done! Output saved to:", CONFIG['output_path'])

if __name__ == "__main__":
    main()

