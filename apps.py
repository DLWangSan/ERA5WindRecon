import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import gc
import json
import psutil
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

def edge_smoothness_loss(pred, lsm_mask):
    B, _, T, H, W = pred.shape
    lsm_mask = F.interpolate(lsm_mask, size=(T, H, W), mode="trilinear", align_corners=False)
    grad_u = torch.abs(pred[:, 0, :, 1:, :] - pred[:, 0, :, :-1, :])  # [B, T, H-1, W]
    grad_v = torch.abs(pred[:, 1, :, :, 1:] - pred[:, 1, :, :, :-1])  # [B, T, H, W-1]
    loss_u = torch.mean(grad_u * lsm_mask[:, 0, :, :-1, :])
    loss_v = torch.mean(grad_v * lsm_mask[:, 0, :, :, :-1])
    return loss_u + loss_v

def total_loss(pred, target, lsm_mask=None, λ=0.01, μ=0.8, γ=0.2):
    data_loss = F.mse_loss(pred, target)
    u, v = pred[:, 0], pred[:, 1]
    phys_loss = divergence_loss(u, v)
    mag_loss = magnitude_loss(pred, target)
    edge_loss = edge_smoothness_loss(pred, lsm_mask) if lsm_mask is not None else 0.0
    return data_loss + λ * phys_loss + μ * mag_loss + γ * edge_loss, data_loss, phys_loss, mag_loss


class Normalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.view(-1, 1, 1, 1)  # [4, 1, 1, 1]，只对应物理通道
        self.std = std.view(-1, 1, 1, 1)    # [4, 1, 1, 1]，只对应物理通道

    def normalize(self, x):
        # 检查输入张量的维度，确保正确处理批量和通道
        if len(x.shape) != 5:  # 预期 [batch_size, channels, t_in, H_lr, W_lr]
            raise ValueError(f"Expected 5D tensor, got shape {x.shape}")
        batch_size = x.shape[0]
        channels = x.shape[1]
        if channels < 7:  # 预期 7 个通道 (4 物理 + 2 坐标 + 1 lsm)
            raise ValueError(f"Expected at least 7 channels, got {channels}")

        # 提取物理通道 (前 4 个通道)
        x_phys = x[:, :4, :, :, :]  # [batch_size, 4, t_in, H_lr, W_lr]
        x_coord_lsm = x[:, 4:, :, :, :] if channels > 4 else None  # [batch_size, 3, t_in, H_lr, W_lr]

        # 确保 self.mean 和 self.std 的形状与 x_phys 的通道数匹配
        if self.mean.shape[0] != 4 or self.std.shape[0] != 4:
            raise ValueError(
                f"Expected self.mean and self.std to have 4 channels, got {self.mean.shape[0]} and {self.std.shape[0]}")

        # 广播 self.mean 和 self.std 到 [batch_size, 4, 1, 1, 1]
        mean_broadcast = self.mean.view(1, 4, 1, 1, 1).expand(batch_size, -1, x.shape[2], x.shape[3], x.shape[4])
        std_broadcast = self.std.view(1, 4, 1, 1, 1).expand(batch_size, -1, x.shape[2], x.shape[3], x.shape[4])

        x_phys_norm = (x_phys - mean_broadcast) / (std_broadcast + 1e-6)  # 只归一化物理通道
        if x_coord_lsm is not None:
            return torch.cat([x_phys_norm, x_coord_lsm], dim=1)  # 拼接回原始通道
        return x_phys_norm

    def denormalize(self, x):
        x_phys = x[:2]  # 只对 u, v 通道反归一化
        x_rest = x[2:] if x.shape[0] > 2 else None
        x_phys_denorm = x_phys * self.std[:2] + self.mean[:2]
        return torch.cat([x_phys_denorm, x_rest], dim=0) if x_rest is not None else x_phys_denorm

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



def compute_mean_std(dataset):
    means, stds = [], []
    for i in tqdm(range(len(dataset))):
        x, _, _ = dataset[i]
        x_phys = x[:4]  # 只取物理通道 (u10, v10, msl, t2m)
        means.append(x_phys.mean(dim=(1, 2, 3)))
        stds.append(x_phys.std(dim=(1, 2, 3)))
    return torch.stack(means).mean(dim=0), torch.stack(stds).mean(dim=0)


def check_model_output_shape(model, dataloader, normalizer, device='cuda'):
    model.eval()
    with torch.no_grad():
        xb, yb, lsm = next(iter(dataloader))
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
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss_epoch = 0
            total_data_loss = 0
            total_phys_loss = 0
            total_mag_loss = 0

            for xb, yb, lsm in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                xb = normalizer.normalize(xb.to(device))
                yb = yb.to(device)
                lsm = lsm.to(device).unsqueeze(1)  # [B, 1, T, H, W]
                pred = model(xb)
                loss, data_loss, phys_loss, mag_loss = total_loss(pred, yb, lsm)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss_epoch += loss.item()
                total_data_loss += data_loss.item()
                total_phys_loss += phys_loss.item()
                total_mag_loss += mag_loss.item()

                del xb, yb, pred, loss  # 清除引用
                torch.cuda.empty_cache()
                gc.collect()  # 手动垃圾回收

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
                        del xb, yb, pred, loss  # 清除引用
                        torch.cuda.empty_cache()
                        gc.collect()  # 手动垃圾回收

                avg_val_loss = val_loss_total / len(val_loader)
                val_losses.append(avg_val_loss)
                print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}")

            with open(loss_log_path, "a") as f:
                f.write(
                    f"{epoch + 1},{avg_train_loss:.6f},{avg_val_loss if avg_val_loss is not None else ''},{avg_data_loss:.6f},{avg_phys_loss:.6f},{avg_mag_loss:.6f}\n")

            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch + 1
                }, model_path)
                print(f"✅ 模型已保存: {model_path}")

            print(f"[Memory] {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")
    except Exception as e:
        print(e.__str__())
        import traceback
        traceback.print_exc()

    plot_losses(train_losses, val_losses, data_losses, phys_losses, mag_losses, save_path="training_loss.png")

    with open("summary.txt", "w") as f:
        f.write(f"Best Train Loss: {min(train_losses):.6f}\n")
        if val_losses:
            f.write(f"Best Val Loss: {min(val_losses):.6f}\n")
            f.write(f"Best Val Epoch: {val_losses.index(min(val_losses)) + 1}\n")


def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='stsr_best.pth')
    parser.add_argument('--era5_path', type=str, default=r'E:\SYSU\lx')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    nc_path = f"{args.era5_path}/era5.nc"
    dataset = ERA5WindSRDataset(
        nc_path=nc_path,
        t_in=6, t_out=36, t_scale=6, s_scale=2,
        use_coord=True
    )

    # 判断 normalizer.json 是否存在
    if not os.path.exists("normalizer.json"):
        mean, std = compute_mean_std(dataset)
        normalizer = Normalizer(mean, std).to(device)
        normalizer.save("normalizer.json")
        print("New mean:", mean, "New std:", std)
    else:
        # 加载已存在的 normalizer.json
        with open("normalizer.json", "r") as f:
            d = json.load(f)
        mean = torch.tensor(d["mean"]).view(-1, 1, 1, 1)  # [4, 1, 1, 1]
        std = torch.tensor(d["std"]).view(-1, 1, 1, 1)  # [4, 1, 1, 1]
        normalizer = Normalizer(mean, std).to(device)
        print("Loaded mean:", mean, "Loaded std:", std)


    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # mean, std = compute_mean_std(train_set)
    # normalizer = Normalizer(mean, std).to(device)
    # normalizer.save("normalizer.json")

    model = STSRNetPlus(t_scale=6, s_scale=2, extra_scale=2.5, c_in=7, use_coord=True).to(device)

    check_model_output_shape(model, train_loader, normalizer, device)

    train_model(model, train_loader, normalizer, val_loader,
                num_epochs=args.epochs, lr=args.lr, lmbd=0.01,
                device=device, model_path=args.model_path, resume=args.resume)


if __name__ == "__main__":
    run_training()
