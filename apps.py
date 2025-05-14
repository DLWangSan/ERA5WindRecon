import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt

from ERA5WindSRDataset import ERA5WindSRDataset
from models import STSRNetPlus


def divergence_loss(u, v):
    dy_u = u[:, :, 1:, :] - u[:, :, :-1, :]
    dx_v = v[:, :, :, 1:] - v[:, :, :, :-1]
    div = dx_v[:, :, :-1, :] + dy_u[:, :, :, :-1]
    return torch.mean(div ** 2)


def magnitude_loss(pred, target):
    mag_pred = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2)
    mag_true = torch.sqrt(target[:, 0]**2 + target[:, 1]**2)
    return F.l1_loss(mag_pred, mag_true)


def total_loss(pred, target, λ=0.01, μ=0.2):
    data_loss = F.mse_loss(pred, target)
    u, v = pred[:, 0], pred[:, 1]
    phys_loss = divergence_loss(u, v)
    mag_loss = magnitude_loss(pred, target)
    return data_loss + λ * phys_loss + μ * mag_loss, data_loss, phys_loss, mag_loss


class Normalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.view(-1, 1, 1, 1)
        self.std = std.view(-1, 1, 1, 1)

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-6)

    def denormalize(self, x):
        return x * self.std + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def save(self, path="normalizer.json"):
        import json
        d = {
            "mean": self.mean.squeeze().cpu().tolist(),
            "std": self.std.squeeze().cpu().tolist()
        }
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)



def compute_mean_std(dataset, num_batches=10):
    means, stds = [], []
    for i in range(num_batches):
        x, _ = dataset[i]
        means.append(x.mean(dim=(1, 2, 3)))
        stds.append(x.std(dim=(1, 2, 3)))
    return torch.stack(means).mean(dim=0), torch.stack(stds).mean(dim=0)


def check_model_output_shape(model, dataloader, normalizer, device='cuda'):
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(dataloader))
        xb = normalizer.normalize(xb.to(device))
        yb = yb.to(device)
        pred = model(xb)
        print("✅ 输入 xb:", xb.shape)
        print("✅ 标签 yb:", yb.shape)
        print("✅ 预测 pred:", pred.shape)
        if pred.shape == yb.shape:
            print("🎉 输出尺寸与标签完全一致，准备训练！")
        else:
            print("❌ 尺寸不一致，请检查模型参数或标签构造逻辑！")


def plot_losses(train_losses, val_losses=None, data_losses=None, phys_losses=None, mag_losses=None, save_path=None):
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

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=9)

    plt.title("Training Loss Curve")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def train_model(model, train_loader, normalizer, val_loader=None,
                num_epochs=50, lr=1e-4, lmbd=0.01, device='cuda',
                model_path='checkpoint.pth', resume=False):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    best_loss = float('inf')
    train_losses, val_losses = [], []
    data_losses, phys_losses, mag_losses = [], [], []
    start_epoch = 0

    if resume and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        print(f"🔄 已加载模型: {model_path}，从第 {start_epoch+1} 轮继续训练")

    loss_log_path = "loss_log.csv"
    need_header = not os.path.exists(loss_log_path) or os.stat(loss_log_path).st_size == 0
    with open(loss_log_path, "a") as f:
        if need_header:
            f.write("epoch,train_loss,val_loss,data_loss,phys_loss,mag_loss\n")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss_epoch = 0
        total_data_loss = 0
        total_phys_loss = 0
        total_mag_loss = 0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            xb = normalizer.normalize(xb.to(device))
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss, data_loss, phys_loss, mag_loss = total_loss(pred, yb, lmbd)
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
                for xb, yb in val_loader:
                    xb = normalizer.normalize(xb.to(device))
                    yb = yb.to(device)
                    pred = model(xb)
                    loss, _, _, _ = total_loss(pred, yb, lmbd)
                    val_loss_total += loss.item()
            avg_val_loss = val_loss_total / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        with open(loss_log_path, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss if avg_val_loss is not None else ''},{avg_data_loss:.6f},{avg_phys_loss:.6f},{avg_mag_loss:.6f}\n")

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch + 1
            }, model_path)
            print(f"✅ 模型已保存: {model_path}")

    plot_losses(train_losses, val_losses, data_losses, phys_losses, mag_losses, save_path="training_loss.png")

    with open("summary.txt", "w") as f:
        f.write(f"Best Train Loss: {min(train_losses):.6f}\n")
        if val_losses:
            f.write(f"Best Val Loss: {min(val_losses):.6f}\n")
            f.write(f"Best Val Epoch: {val_losses.index(min(val_losses)) + 1}\n")


def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='stsr_best.pth')
    parser.add_argument('--era5_path', type=str, default=r'E:\SYSU\lx')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nc_path = f"{args.era5_path}/era5.nc"
    dataset = ERA5WindSRDataset(
        nc_path=nc_path,
        t_in=6, t_out=36, t_scale=6, s_scale=4,
        use_coord=True
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    mean, std = compute_mean_std(train_set)
    normalizer = Normalizer(mean, std).to(device)
    normalizer.save("normalizer.json")

    model = STSRNetPlus(t_scale=6, s_scale=4, extra_scale=2.5, c_in=7, use_coord=True).to(device)

    check_model_output_shape(model, train_loader, normalizer, device)

    train_model(model, train_loader, normalizer, val_loader,
                num_epochs=args.epochs, lr=args.lr, lmbd=0.01,
                device=device, model_path=args.model_path, resume=args.resume)


if __name__ == "__main__":
    run_training()
