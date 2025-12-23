import torch
import xarray as xr
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import argparse

from dataset import ERA5WindSRDataset
from models import STSRNetPlus, STSRNetPlusCompare1, STSRNetPlusCompare2, STSRNetPlusCompare3, STSRNetPlusCompare4
from train import Normalizer


def load_model(model_type, checkpoint_path, t_scale=12, s_scale=2, extra_scale=2.5, 
               c_in=7, use_coord=True, use_lsm=True, device='cuda'):
    """Load model and weights."""
    if model_type == "normal":
        model = STSRNetPlus(t_scale=t_scale, s_scale=s_scale, extra_scale=extra_scale, 
                           c_in=c_in, use_coord=use_coord, use_lsm=use_lsm)
    elif model_type == "compare1":
        model = STSRNetPlusCompare1(t_scale=t_scale, s_scale=s_scale, extra_scale=extra_scale, 
                                    c_in=c_in, use_coord=use_coord, use_lsm=use_lsm)
    elif model_type == "compare2":
        model = STSRNetPlusCompare2(t_scale=t_scale, s_scale=s_scale, extra_scale=extra_scale, 
                                    c_in=c_in, use_coord=use_coord, use_lsm=use_lsm)
    elif model_type == "compare3":
        model = STSRNetPlusCompare3(t_scale=t_scale, s_scale=s_scale, extra_scale=extra_scale, 
                                    c_in=c_in, use_coord=use_coord, use_lsm=use_lsm)
    elif model_type == "compare4":
        model = STSRNetPlusCompare4(t_scale=t_scale, s_scale=s_scale, extra_scale=extra_scale, 
                                    c_in=c_in, use_coord=use_coord, use_lsm=use_lsm)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    return model


def load_normalizer(normalizer_path, device='cuda'):
    """Load normalizer."""
    with open(normalizer_path, 'r') as f:
        d = json.load(f)
    mean = torch.tensor(d["mean"]).view(-1, 1, 1, 1).to(device)
    std = torch.tensor(d["std"]).view(-1, 1, 1, 1).to(device)
    return Normalizer(mean, std).to(device)


def generate_reconstructed_nc(model, dataset, normalizer, output_path, batch_size=8, device='cuda'):
    """Generate reconstructed NetCDF file using the model."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ds = dataset.ds
    t_in, t_out, t_scale, s_scale, extra_scale = dataset.t_in, dataset.t_out, dataset.t_scale, dataset.s_scale, 2.5
    H_full, W_full = dataset.H_full, dataset.W_full
    H_lr, W_lr = dataset.H // s_scale, dataset.W // s_scale
    H_hr = int(H_lr * s_scale * extra_scale)
    W_hr = int(W_lr * s_scale * extra_scale)
    t_hr = t_out * 6

    time_start = ds['valid_time'].values[0]
    time_end = ds['valid_time'].values[-1]
    time_step = np.timedelta64(10, 'm')
    total_hours = int((time_end - time_start) / np.timedelta64(1, 'h'))
    total_samples = total_hours * 6 + 1
    times = [time_start + i * time_step for i in range(total_samples)]

    u10_recon = np.zeros((total_samples, H_hr, W_hr), dtype=np.float32)
    v10_recon = np.zeros((total_samples, H_hr, W_hr), dtype=np.float32)
    count = np.zeros((total_samples, H_hr, W_hr), dtype=np.float32)

    with torch.no_grad():
        for i, (xb, _, lsm) in enumerate(tqdm(dataloader, desc="Generating reconstructed data")):
            xb = normalizer.normalize(xb.to(device))
            lsm = lsm.to(device).unsqueeze(1)
            pred = model(xb)
            pred = pred.cpu().numpy()

            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(dataset))
            for j in range(batch_end - batch_start):
                sample_idx = batch_start + j
                sample_start_idx = dataset.valid_start[sample_idx]
                time_offset = int((ds['valid_time'].values[sample_start_idx] - time_start) / time_step)
                t_start = time_offset
                t_end = t_start + t_hr
                if t_end <= total_samples:
                    u10_recon[t_start:t_end] += pred[j, 0]
                    v10_recon[t_start:t_end] += pred[j, 1]
                    count[t_start:t_end] += 1
                else:
                    valid_steps = total_samples - t_start
                    u10_recon[t_start:total_samples] += pred[j, 0, :valid_steps]
                    v10_recon[t_start:total_samples] += pred[j, 1, :valid_steps]
                    count[t_start:total_samples] += 1

            torch.cuda.empty_cache()

    count = np.maximum(count, 1)
    u10_recon /= count
    v10_recon /= count

    lat_lr = ds['latitude'].values[:dataset.H][::s_scale]
    lon_lr = ds['longitude'].values[:dataset.W][::s_scale]
    lat_hr = np.linspace(lat_lr[0], lat_lr[-1], H_hr)
    lon_hr = np.linspace(lon_lr[0], lon_lr[-1], W_hr)

    recon_ds = xr.Dataset(
        {
            'u10': (['valid_time', 'latitude', 'longitude'], u10_recon),
            'v10': (['valid_time', 'latitude', 'longitude'], v10_recon),
        },
        coords={
            'valid_time': times,
            'latitude': lat_hr,
            'longitude': lon_hr,
        }
    )

    recon_ds['u10'].attrs = {'units': 'm s**-1', 'long_name': '10m u-component of wind'}
    recon_ds['v10'].attrs = {'units': 'm s**-1', 'long_name': '10m v-component of wind'}
    recon_ds.attrs = {
        'description': 'Reconstructed high-resolution wind fields using STSRNet',
        'model_type': model.__class__.__name__,
        't_scale': t_scale,
        's_scale': s_scale,
        'extra_scale': extra_scale,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    recon_ds.to_netcdf(output_path)
    print(f"Reconstructed NetCDF file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate reconstructed NetCDF file using trained model")
    parser.add_argument('--model_type', type=str, default='normal',
                        choices=['normal', 'compare1', 'compare2', 'compare3', 'compare4'], 
                        help='Model type')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--normalizer_path', type=str, default='normalizer.json', 
                        help='Path to normalizer file')
    parser.add_argument('--era5_path', type=str, required=True,
                        help='Path to ERA5 NetCDF file')
    parser.add_argument('--lsm_path', type=str, default='lsm_era5.nc', 
                        help='Path to land-sea mask NetCDF file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output NetCDF file path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--t_in', type=int, default=18, help='Input time steps')
    parser.add_argument('--t_out', type=int, default=36, help='Output time steps')
    parser.add_argument('--t_scale', type=int, default=2, help='Temporal scale factor')
    parser.add_argument('--s_scale', type=int, default=2, help='Spatial scale factor')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    dataset = ERA5WindSRDataset(
        nc_path=args.era5_path,
        t_in=args.t_in,
        t_out=args.t_out,
        t_scale=args.t_scale,
        s_scale=args.s_scale,
        use_coord=True,
        lsm_path=args.lsm_path
    )

    model = load_model(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        t_scale=12,
        s_scale=args.s_scale,
        extra_scale=2.5,
        c_in=7,
        use_coord=True,
        use_lsm=True,
        device=device
    )
    normalizer = load_normalizer(args.normalizer_path, device)

    generate_reconstructed_nc(
        model=model,
        dataset=dataset,
        normalizer=normalizer,
        output_path=args.output_path,
        batch_size=args.batch_size,
        device=device
    )


if __name__ == "__main__":
    main()

