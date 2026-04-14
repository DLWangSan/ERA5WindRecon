import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import json
import argparse
import matplotlib.pyplot as plt

from dataset import ERA5WindSRDataset
from models import STSRNetPlus, STSRNetPlusCompare1, STSRNetPlusCompare2, STSRNetPlusCompare3, STSRNetPlusCompare4


def divergence_loss(u, v):
    """Compute horizontal divergence loss for mass conservation."""
    dy_u = u[:, :, 1:, :] - u[:, :, :-1, :]
    dx_v = v[:, :, :, 1:] - v[:, :, :, :-1]
    div = dx_v[:, :, :-1, :] + dy_u[:, :, :, :-1]
    return torch.mean(div ** 2)


def magnitude_loss(pred, target):
    """Compute wind speed magnitude loss."""
    mag_pred = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2)
    mag_true = torch.sqrt(target[:, 0]**2 + target[:, 1]**2)
    return F.l1_loss(mag_pred, mag_true)


def edge_smoothness_loss(pred, lsm_mask):
    """Compute edge smoothness loss weighted by land-sea mask."""
    B, _, T, H, W = pred.shape
    lsm_mask = F.interpolate(lsm_mask, size=(T, H, W), mode="trilinear", align_corners=False)
    grad_u = torch.abs(pred[:, 0, :, 1:, :] - pred[:, 0, :, :-1, :])
    grad_v = torch.abs(pred[:, 1, :, :, 1:] - pred[:, 1, :, :, :-1])
    loss_u = torch.mean(grad_u * lsm_mask[:, 0, :, :-1, :])
    loss_v = torch.mean(grad_v * lsm_mask[:, 0, :, :, :-1])
    return loss_u + loss_v


def total_loss(pred, target, lsm_mask=None, λ=0.01, μ=0.8, γ=0.2):
    """Total loss combining data fidelity and physics constraints."""
    data_loss = F.mse_loss(pred, target)
    u, v = pred[:, 0], pred[:, 1]
    phys_loss = divergence_loss(u, v)
    mag_loss = magnitude_loss(pred, target)
    edge_loss = edge_smoothness_loss(pred, lsm_mask) if lsm_mask is not None else 0.0
    return data_loss + λ * phys_loss + μ * mag_loss + γ * edge_loss, data_loss, phys_loss, mag_loss


class Normalizer:
    """Normalizer for input variables."""
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.view(-1, 1, 1, 1)
        self.std = std.view(-1, 1, 1, 1)

    def normalize(self, x):
        if len(x.shape) != 5:
            raise ValueError(f"Expected 5D tensor, got shape {x.shape}")
        batch_size = x.shape[0]
        channels = x.shape[1]
        if channels < 7:
            raise ValueError(f"Expected at least 7 channels, got {channels}")

        x_phys = x[:, :4, :, :, :]
        x_coord_lsm = x[:, 4:, :, :, :] if channels > 4 else None

        if self.mean.shape[0] != 4 or self.std.shape[0] != 4:
            raise ValueError(f"Expected 4 channels for mean/std, got {self.mean.shape[0]}")

        mean_broadcast = self.mean.view(1, 4, 1, 1, 1).expand(batch_size, -1, x.shape[2], x.shape[3], x.shape[4])
        std_broadcast = self.std.view(1, 4, 1, 1, 1).expand(batch_size, -1, x.shape[2], x.shape[3], x.shape[4])

        x_phys_norm = (x_phys - mean_broadcast) / (std_broadcast + 1e-6)
        if x_coord_lsm is not None:
            return torch.cat([x_phys_norm, x_coord_lsm], dim=1)
        return x_phys_norm

    def denormalize(self, x):
        x_phys = x[:2]
        x_rest = x[2:] if x.shape[0] > 2 else None
        x_phys_denorm = x_phys * self.std[:2] + self.mean[:2]
        return torch.cat([x_phys_denorm, x_rest], dim=0) if x_rest is not None else x_phys_denorm

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def save(self, path="normalizer.json"):
        d = {
            "mean": self.mean.squeeze().cpu().tolist(),
            "std": self.std.squeeze().cpu().tolist()
        }
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)


def compute_mean_std(dataset):
    """Compute mean and std statistics for normalization."""
    means, stds = [], []
    for i in tqdm(range(len(dataset))):
        x, _, _ = dataset[i]
        x_phys = x[:4]
        means.append(x_phys.mean(dim=(1, 2, 3)))
        stds.append(x_phys.std(dim=(1, 2, 3)))
    return torch.stack(means).mean(dim=0), torch.stack(stds).mean(dim=0)


def plot_losses(train_losses, val_losses=None, data_losses=None, phys_losses=None, mag_losses=None, save_path=None):
    """Plot training loss curves."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(epochs, train_losses, label='Total Loss', color='tab:blue', linewidth=2)
    if val_losses:
        ax1.plot(epochs, val_losses, label='Val Loss', color='tab:orange', linewidth=2)
    if data_losses:
        ax1.plot(epochs, data_losses, label='Data Loss', color='tab:green', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total / Data Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    if phys_losses:
        ax2.plot(epochs, phys_losses, label='Divergence Loss', color='tab:red', linestyle='-.')
    if mag_losses:
        ax2.plot(epochs, mag_losses, label='Magnitude Loss', color='tab:purple', linestyle='-.')
    ax2.set_ylabel('Physics / Mag Loss', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=9)

    plt.title("Training Loss Curve")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def train_model(model, train_loader, normalizer, val_loader=None,
                num_epochs=50, lr=1e-4, device='cuda',
                pth_name='checkpoint.pth', resume=False,
                loss_log_path="loss_log.csv", save_path=""):

    if not os.path.exists(os.path.dirname(loss_log_path)):
        os.makedirs(os.path.dirname(loss_log_path))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    best_loss = float('inf')
    train_losses, val_losses = [], []
    data_losses, phys_losses, mag_losses = [], [], []
    start_epoch = 0

    if resume and os.path.exists(os.path.join(save_path, pth_name)):
        checkpoint = torch.load(os.path.join(save_path, pth_name))
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed training from epoch {start_epoch+1}")

    need_header = not os.path.exists(loss_log_path) or os.stat(loss_log_path).st_size == 0
    with open(loss_log_path, "a") as f:
        if need_header:
            f.write("epoch,train_loss,val_loss,data_loss,phys_loss,mag_loss\n")
    
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss_epoch = 0
            total_data_loss = 0
            total_phys_loss = 0
            total_mag_loss = 0

            for xb, yb, lsm in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                xb = normalizer.normalize(xb.to(device, non_blocking=True))
                yb = yb.to(device, non_blocking=True)
                lsm = lsm.to(device).unsqueeze(1)
                pred = model(xb)
                loss, data_loss, phys_loss, mag_loss = total_loss(pred, yb, lsm)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss_epoch += loss.item()
                total_data_loss += data_loss.item()
                total_phys_loss += phys_loss.item()
                total_mag_loss += mag_loss.item()

            scheduler.step()
            avg_train_loss = total_loss_epoch / len(train_loader)
            avg_data_loss = total_data_loss / len(train_loader)
            avg_phys_loss = total_phys_loss / len(train_loader)
            avg_mag_loss = total_mag_loss / len(train_loader)

            train_losses.append(avg_train_loss)
            data_losses.append(avg_data_loss)
            phys_losses.append(avg_phys_loss)
            mag_losses.append(avg_mag_loss)

            avg_val_loss = None
            if val_loader:
                model.eval()
                val_loss_total = 0
                with torch.no_grad():
                    for xb, yb, lsm in val_loader:
                        xb = normalizer.normalize(xb.to(device))
                        yb = yb.to(device)
                        lsm = lsm.to(device).unsqueeze(1)
                        pred = model(xb)
                        loss, _, _, _ = total_loss(pred, yb, lsm)
                        val_loss_total += loss.item()

                avg_val_loss = val_loss_total / len(val_loader)
                val_losses.append(avg_val_loss)
                print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}")

            with open(loss_log_path, "a") as f:
                f.write(f"{epoch + 1},{avg_train_loss:.6f},{avg_val_loss if avg_val_loss is not None else ''},{avg_data_loss:.6f},{avg_phys_loss:.6f},{avg_mag_loss:.6f}\n")

            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch + 1
                }, os.path.join(save_path, pth_name))
                print(f"Model saved: {os.path.join(save_path, pth_name)}")

    except Exception as e:
        print(e.__str__())
        import traceback
        traceback.print_exc()

    plot_losses(train_losses, val_losses, data_losses, phys_losses, mag_losses,
                save_path=os.path.join(save_path, "training_loss.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pth_name', type=str, default='stsr_best.pth')
    parser.add_argument('--era5_path', type=str, required=True)
    parser.add_argument('--filename', type=str, default='ERA5_2020-2024.nc')
    parser.add_argument('--model_type', type=str, default='normal', 
                        choices=['normal', 'compare1', 'compare2', 'compare3', 'compare4'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lsm_path', type=str, default='lsm_era5.nc')
    parser.add_argument('--t_in', type=int, default=18, help='Input time steps (must fit short example series)')
    parser.add_argument('--t_out', type=int, default=36, help='Output time steps (must be <= number of time steps in NetCDF)')
    args = parser.parse_args()

    save_path = os.path.join("./runs/train", f"{args.model_type}")
    os.makedirs(save_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    nc_path = f"{args.era5_path}/{args.filename}"

    dataset = ERA5WindSRDataset(
        nc_path=nc_path,
        t_in=args.t_in, t_out=args.t_out, t_scale=2, s_scale=2,
        use_coord=True, lsm_path=args.lsm_path
    )

    if not os.path.exists("normalizer.json"):
        mean, std = compute_mean_std(dataset)
        normalizer = Normalizer(mean, std).to(device)
        normalizer.save("normalizer.json")
        print("Computed mean:", mean, "std:", std)
    else:
        with open("normalizer.json", "r") as f:
            d = json.load(f)
        mean = torch.tensor(d["mean"]).view(-1, 1, 1, 1)
        std = torch.tensor(d["std"]).view(-1, 1, 1, 1)
        normalizer = Normalizer(mean, std).to(device)
        print("Loaded mean:", mean, "std:", std)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    if args.model_type == "normal":
        model = STSRNetPlus(t_scale=12, s_scale=2, extra_scale=2.5, c_in=7, use_coord=True).to(device)
    elif args.model_type == "compare1":
        model = STSRNetPlusCompare1(t_scale=12, s_scale=2, extra_scale=2.5, c_in=7, use_coord=True).to(device)
    elif args.model_type == "compare2":
        model = STSRNetPlusCompare2(t_scale=12, s_scale=2, extra_scale=2.5, c_in=7, use_coord=True).to(device)
    elif args.model_type == "compare3":
        model = STSRNetPlusCompare3(t_scale=12, s_scale=2, extra_scale=2.5, c_in=7, use_coord=True).to(device)
    elif args.model_type == "compare4":
        model = STSRNetPlusCompare4(t_scale=12, s_scale=2, extra_scale=2.5, c_in=7, use_coord=True).to(device)
    else:
        print(f"Unknown model type: {args.model_type}")
        return

    train_model(model, train_loader, normalizer, val_loader,
                num_epochs=args.epochs, lr=args.lr,
                device=device, pth_name=args.pth_name, resume=args.resume,
                loss_log_path=os.path.join(save_path, "train_loss_log.csv"),
                save_path=save_path)


if __name__ == "__main__":
    main()






