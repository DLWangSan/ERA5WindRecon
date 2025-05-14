# ========================
# 模型结构（UNet-style + 上采样融合）
# ========================
import torch
from torch import nn
import xarray as xr
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
from torch.utils.data import Dataset


def generate_coord_tensor(B, T, H, W, device):
    """生成标准化坐标张量: [B, 2, T, H, W]"""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W), indexing='ij')
    coords = torch.stack([xx, yy], dim=0)  # [2, H, W]
    coords = coords.unsqueeze(1).repeat(1, T, 1, 1)  # [2, T, H, W]
    coords = coords.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # [B, 2, T, H, W]
    return coords.to(device)


class ERA5WindSRDataset(Dataset):
    def __init__(self, nc_path, t_in=6, t_out=36, t_scale=6, s_scale=4, use_interp_label=True, use_coord=False):
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

        self.valid_start = list(range(0, self.T - self.t_out + 1))

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

        if self.use_coord:
            lat = torch.tensor(self.ds["latitude"].values[:self.H][::self.s_scale], dtype=torch.float32)
            lon = torch.tensor(self.ds["longitude"].values[:self.W][::self.s_scale], dtype=torch.float32)

            yy, xx = torch.meshgrid(lat, lon, indexing='ij')  # [H_lr, W_lr]

            # print("🌐 真实经纬度网格:")
            # print("lat shape:", yy.shape, "lon shape:", xx.shape)
            # print("lat range:", yy.min().item(), "to", yy.max().item())
            # print("lon range:", xx.min().item(), "to", xx.max().item())

            yy = 2 * (yy - yy.min()) / (yy.max() - yy.min()) - 1  # → [-1, 1]
            xx = 2 * (xx - xx.min()) / (xx.max() - xx.min()) - 1

            coord = torch.stack([xx, yy], dim=0)  # [2, H_lr, W_lr]
            coord = coord.unsqueeze(1).repeat(1, self.t_in, 1, 1)  # [2, T_in, H_lr, W_lr]
            x = torch.cat([x, coord], dim=0)  # 增加坐标通道



        if self.use_interp_label:
            t_src = np.arange(0, self.t_out, self.t_scale)
            t_dst = np.arange(0, self.t_out)

            def interp_time_space(var):
                f = interp1d(t_src, var[::self.t_scale], axis=0, kind='linear', fill_value='extrapolate')
                var_t = f(t_dst)
                return zoom(var_t, zoom=(1, 2.5, 2.5), order=1)

            u_interp = interp_time_space(self.u10[t0:t0 + self.t_out])
            v_interp = interp_time_space(self.v10[t0:t0 + self.t_out])
            y = torch.tensor(np.stack([u_interp, v_interp], axis=0))
        else:
            y = torch.tensor(np.stack([u_hr, v_hr], axis=0))


        return x.float(), y.float()


