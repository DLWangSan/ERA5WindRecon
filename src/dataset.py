import torch
import xarray as xr
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
from torch.utils.data import Dataset


class ERA5WindSRDataset(Dataset):
    """Dataset for ERA5 wind field super-resolution task."""
    
    def __init__(self, nc_path, t_in=18, t_out=36, t_scale=2, s_scale=2, 
                 use_interp_label=True, use_coord=True, lsm_path="lsm_era5.nc"):
        self.ds = xr.open_dataset(nc_path)
        self.u10 = self.ds["u10"].values.astype(np.float32)
        self.v10 = self.ds["v10"].values.astype(np.float32)
        self.msl = self.ds["msl"].values.astype(np.float32)
        self.t2m = self.ds["t2m"].values.astype(np.float32)

        self.t_in = t_in
        self.t_out = t_out
        self.t_scale = t_scale
        self.s_scale = s_scale
        self.use_interp_label = use_interp_label
        self.use_coord = use_coord

        self.T, self.H_full, self.W_full = self.u10.shape
        self.H = (self.H_full // s_scale) * s_scale
        self.W = (self.W_full // s_scale) * s_scale

        self.u10 = self.u10[:, :self.H, :self.W]
        self.v10 = self.v10[:, :self.H, :self.W]
        self.msl = self.msl[:, :self.H, :self.W]
        self.t2m = self.t2m[:, :self.H, :self.W]

        if lsm_path:
            lsm_ds = xr.open_dataset(lsm_path)
            lsm_data = lsm_ds['lsm'].squeeze().values.astype(np.float32)
            lsm_lat = lsm_ds['latitude'].values
            lsm_lon = lsm_ds['longitude'].values
            era_lat = self.ds['latitude'].values[:self.H]
            era_lon = self.ds['longitude'].values[:self.W]
            lat_min, lat_max = era_lat.min(), era_lat.max()
            lon_min, lon_max = era_lon.min(), era_lon.max()
            lat_idx = (lsm_lat <= lat_max) & (lsm_lat >= lat_min)
            lon_idx = (lsm_lon >= lon_min) & (lsm_lon <= lon_max)
            lsm_sub = lsm_data[np.where(lat_idx)[0][0]:np.where(lat_idx)[0][-1] + 1,
                             np.where(lon_idx)[0][0]:np.where(lon_idx)[0][-1] + 1]
            if era_lat[0] < era_lat[-1]:
                lsm_sub = np.flip(lsm_sub, axis=0)
            self.lsm = lsm_sub
        else:
            self.lsm = None

        self.valid_start = list(range(0, self.T - self.t_out + 1, t_in // t_scale))

    def __len__(self):
        return len(self.valid_start)

    def __getitem__(self, idx):
        t0 = self.valid_start[idx]
        H_lr, W_lr = self.H // self.s_scale, self.W // self.s_scale

        def downsample(var):
            var_hr = var[t0:t0 + self.t_out]
            var_lr = var_hr[::self.t_scale]
            var_lr = torch.tensor(var_lr).unsqueeze(1)
            var_lr = F.interpolate(var_lr, size=(H_lr, W_lr), mode="bilinear", align_corners=False).squeeze(1)
            return var_hr, var_lr

        u_hr, u_lr = downsample(self.u10)
        v_hr, v_lr = downsample(self.v10)
        msl_hr, msl_lr = downsample(self.msl)
        t2m_hr, t2m_lr = downsample(self.t2m)

        x = torch.stack([u_lr, v_lr, msl_lr, t2m_lr], dim=0)
        n_t_lr = u_lr.shape[0]

        if self.use_coord:
            lat = torch.tensor(self.ds["latitude"].values[:self.H][::self.s_scale], dtype=torch.float32)
            lon = torch.tensor(self.ds["longitude"].values[:self.W][::self.s_scale], dtype=torch.float32)
            yy, xx = torch.meshgrid(lat, lon, indexing='ij')
            yy = 2 * (yy - yy.min()) / (yy.max() - yy.min()) - 1
            xx = 2 * (xx - xx.min()) / (xx.max() - xx.min()) - 1
            coord = torch.stack([xx, yy], dim=0)
            coord = coord.unsqueeze(1).repeat(1, n_t_lr, 1, 1)
            x = torch.cat([x, coord], dim=0)

        if self.lsm is not None:
            lsm_lr = torch.tensor(self.lsm).unsqueeze(0).unsqueeze(0)
            lsm_lr = F.interpolate(lsm_lr, size=(H_lr, W_lr), mode="bilinear", align_corners=False)
            lsm_lr = lsm_lr.squeeze(0).squeeze(0)
            lsm_lr = lsm_lr.repeat(n_t_lr, 1, 1)
            x = torch.cat([x, lsm_lr.unsqueeze(0)], dim=0)

        if self.use_interp_label:
            t_src = np.arange(0, self.t_out, self.t_scale)
            t_dst = np.linspace(0, self.t_out - 1, self.t_out * 6)
            def interp_time_space(var):
                f = interp1d(t_src, var[::self.t_scale], axis=0, kind='cubic', fill_value='extrapolate')
                var_t = f(t_dst)
                return zoom(var_t, zoom=(1, 2.5, 2.5), order=1)
            u_interp = interp_time_space(self.u10[t0:t0 + self.t_out])
            v_interp = interp_time_space(self.v10[t0:t0 + self.t_out])
            u_interp = u_interp.astype(np.float32, copy=False)
            v_interp = v_interp.astype(np.float32, copy=False)
            y = torch.tensor(np.stack([u_interp, v_interp], axis=0), dtype=torch.float32)
        else:
            y = torch.tensor(np.stack([u_hr, v_hr], axis=0))

        return x.float(), y.float(), lsm_lr.float()

