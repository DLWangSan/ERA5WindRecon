import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import xarray as xr
from torch.utils.data import DataLoader

from ERA5WindSRDataset import ERA5WindSRDataset
from apps import Normalizer
from models import STSRNetPlus


def visualize_prediction(model, dataloader, normalizer, raw_era5, device='cuda', save_dir="viz"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(dataloader))
        xb = xb.to(device)
        yb = yb.to(device)

        # 仅提取物理通道（不包括坐标）
        xb_phys = xb[:, :4]  # [B, 4, T, H, W]
        xb_coord = xb[:, 4:] if xb.shape[1] > 4 else None

        xb_phys_norm = normalizer.normalize(xb_phys)  # 仅标准化前4通道
        xb_norm = torch.cat([xb_phys_norm, xb_coord], dim=1) if xb_coord is not None else xb_phys_norm

        # 自动添加坐标通道（如模型需要）
        if getattr(model, "use_coord", False):
            B, _, T, H, W = xb.shape
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing='ij')
            coord = torch.stack([xx, yy], dim=0).unsqueeze(0).unsqueeze(2).repeat(B, 1, T, 1, 1)
            xb_norm = torch.cat([xb_norm, coord], dim=1)



        pred = model(xb_norm)
        pred = pred * normalizer.std[:2].to(device) + normalizer.mean[:2].to(device)

        # === 取第一个样本、某一时间点 ===
        idx, t = 0, 5
        t_global = dataloader.dataset.valid_start[idx] + t

        u_raw = raw_era5['u10'][t_global].values.astype(np.float32)
        u_raw = u_raw[:yb.shape[-2], :yb.shape[-1]]  # 裁剪至 50×40

        u_pred = pred[idx, 0, t].cpu().numpy()
        u_true = yb[idx, 0, t].cpu().numpy()
        err = u_pred - u_true

        # === 设置统一 colorbar 范围 ===
        vmin = min(u_raw.min(), u_pred.min(), u_true.min())
        vmax = max(u_raw.max(), u_pred.max(), u_true.max())
        err_vmin, err_vmax = err.min(), err.max()

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        images = [u_raw, u_pred, u_true, err]
        titles = ["ERA5 Original", "Predicted", "Training Label", "Error"]
        cmaps = ['coolwarm'] * 3 + ['bwr']
        vmins = [vmin, vmin, vmin, err_vmin]
        vmaxs = [vmax, vmax, vmax, err_vmax]

        for ax, data, title, cmap, vmin_, vmax_ in zip(axs, images, titles, cmaps, vmins, vmaxs):
            im = ax.imshow(data, cmap=cmap, vmin=vmin_, vmax=vmax_)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=0.7)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/prediction_sample.png", dpi=300)
        plt.close()
        print(f"🎨 可视化图像已保存至 {save_dir}/prediction_sample.png")


def visualize_vector_field(model, dataloader, normalizer, raw_era5, device='cuda', save_dir="viz"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        xb, yb = next(iter(dataloader))
        xb = xb.to(device)
        yb = yb.to(device)

        # 仅标准化前4通道
        xb_phys = xb[:, :4]
        xb_coord = xb[:, 4:] if xb.shape[1] > 4 else None
        xb_phys_norm = normalizer.normalize(xb_phys)
        xb_norm = torch.cat([xb_phys_norm, xb_coord], dim=1) if xb_coord is not None else xb_phys_norm

        # 添加坐标（兼容 use_coord）
        if getattr(model, "use_coord", False):
            B, _, T, H, W = xb.shape
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing='ij')

            coords = torch.stack([xx, yy], dim=0).unsqueeze(0).unsqueeze(2).repeat(B, 1, T, 1, 1)
            xb_norm = torch.cat([xb_norm, coords], dim=1)

        pred = model(xb_norm)
        pred = pred * normalizer.std[:2].to(device) + normalizer.mean[:2].to(device)

        # === 选择第0个样本，第t帧 ===
        idx, t = 7, 5
        t_global = dataloader.dataset.valid_start[idx] + t

        # 原始 ERA5 的分辨率（例如 20×16）
        u_raw = raw_era5['u10'][t_global].values.astype(np.float32)
        v_raw = raw_era5['v10'][t_global].values.astype(np.float32)

        u_pred = pred[idx, 0, t].cpu().numpy()
        v_pred = pred[idx, 1, t].cpu().numpy()

        u_true = yb[idx, 0, t].cpu().numpy()
        v_true = yb[idx, 1, t].cpu().numpy()

        # 可视化函数：绘制箭头
        def plot_uv(ax, u, v, title, step=2, scale=100):
            H, W = u.shape
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            ax.quiver(xx[::step, ::step], yy[::step, ::step],
                      u[::step, ::step], v[::step, ::step], scale=scale)
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.axis('off')

        fig, axs = plt.subplots(1, 4, figsize=(18, 4))
        plot_uv(axs[0], u_raw, v_raw, "ERA5 Original")
        plot_uv(axs[1], u_pred, v_pred, "Predicted")
        plot_uv(axs[2], u_true, v_true, "Training Label")
        plot_uv(axs[3], u_pred - u_true, v_pred - v_true, "Error")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/vector_field.png", dpi=300)
        plt.close()
        print(f"✅ 流场箭头图已保存至 {save_dir}/vector_field.png")


# === 加载 Normalizer 参数 ===
def load_normalizer(json_path="normalizer.json"):
    with open(json_path, "r") as f:
        d = json.load(f)
    mean = torch.tensor(d["mean"]).view(-1, 1, 1, 1)
    std = torch.tensor(d["std"]).view(-1, 1, 1, 1)
    return Normalizer(mean, std)

# === 主流程 ===
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nc_path = "../../lx/era5.nc"

    # 加载数据集和 dataloader
    dataset = ERA5WindSRDataset(nc_path, t_in=6, t_out=36, t_scale=6, s_scale=4, use_interp_label=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # 加载模型
    # model = STSRNet(t_scale=6, s_scale=4, extra_scale=2.5, c_in=4).to(device)
    model = STSRNetPlus(t_scale=6, s_scale=4, extra_scale=2.5, c_in=6, use_coord=True).to(device)
    checkpoint = torch.load("stsr_best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print("✅ 模型加载完成")

    # 加载标准化器
    normalizer = load_normalizer().to(device)
    # ⚠️ 仅保留前4个物理通道（去除 CoordConv 坐标通道的均值和方差）
    normalizer.mean = normalizer.mean[:4]
    normalizer.std = normalizer.std[:4]

    ds_raw = xr.open_dataset("../../lx/era5.nc")
    visualize_prediction(model, dataloader, normalizer, raw_era5=ds_raw)
    visualize_vector_field(model, dataloader, normalizer, raw_era5=ds_raw)


if __name__ == "__main__":
    main()
