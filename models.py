# ========================
# 模型结构（UNet-style + 上采样融合）
# ========================
import torch
from torch import nn
from torchinfo import summary


def generate_coord_tensor(B, T, H, W, device):
    """生成标准化坐标张量: [B, 2, T, H, W]"""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W), indexing='ij')
    coords = torch.stack([xx, yy], dim=0)  # [2, H, W]
    coords = coords.unsqueeze(1).repeat(1, T, 1, 1)  # [2, T, H, W]
    coords = coords.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # [B, 2, T, H, W]
    return coords.to(device)


# ========================
# A Baseline Model STSRNet
# ========================
class STSRNet(nn.Module):
    def __init__(self, t_scale=6, s_scale=4, extra_scale=2.5, c_in=4, use_coord=False):
        super().__init__()
        self.t_scale = t_scale
        self.total_spatial_scale = s_scale * extra_scale
        self.use_coord = use_coord
        self.c_in = c_in + 2 if use_coord else c_in

        self.encoder = nn.Sequential(
            nn.Conv3d(self.c_in, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv3d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.Conv3d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm3d(128),
            ) for _ in range(4)]
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv3d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.temporal_up = nn.Upsample(scale_factor=(t_scale, 1, 1), mode='trilinear', align_corners=False)

        self.spatial_up = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(1, self.total_spatial_scale, self.total_spatial_scale), mode='trilinear', align_corners=False),
            nn.Conv3d(64, 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        if self.use_coord:
            B, _, T, H, W = x.shape
            coord = generate_coord_tensor(B, T, H, W, x.device)
            x = torch.cat([x, coord], dim=1)

        x = self.encoder(x)
        res = x
        for block in self.res_blocks:
            res = res + block(res)
        attn = self.spatial_attention(res)
        res = res * attn
        x = self.temporal_up(res)
        x = self.spatial_up(x)
        return x


# ========================
# 模型结构（Physics-aware + Cross-scale Transformer + Residual + CoordConv）
# ========================

class STSRNetPlus(nn.Module):
    def __init__(self, t_scale=6, s_scale=4, extra_scale=2.5, c_in=4, use_coord=True):
        super().__init__()
        self.t_scale = t_scale
        self.total_spatial_scale = s_scale * extra_scale
        self.use_coord = use_coord
        self.base_c_in = c_in
        self.c_in = c_in + 2 if use_coord else c_in

        self.encoder = nn.Sequential(
            nn.Conv3d(self.c_in, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fusion_dilated = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )
        self.fusion_res = nn.Conv3d(128, 128, kernel_size=1)

        encoder_layer_1 = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256)
        encoder_layer_2 = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512)
        self.temporal_transformer = nn.ModuleList([encoder_layer_1, encoder_layer_2])

        self.attn_weight = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )

        self.physics_gate = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv3d(128, 128, kernel_size=1)
        )

        self.temporal_up = nn.Upsample(scale_factor=(t_scale, 1, 1), mode='trilinear', align_corners=False)

        self.spatial_up = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, self.total_spatial_scale, self.total_spatial_scale), mode='trilinear', align_corners=False),
            nn.Conv3d(64, 2, kernel_size=3, padding=1),
        )

        self.attn_maps = []
        def hook_fn(m, i, o):
            if isinstance(o, tuple):
                o = o[0]
            self.attn_maps.append(o.detach().cpu())
        for layer in self.temporal_transformer:
            layer.self_attn.register_forward_hook(hook_fn)

    def forward(self, x):
        if self.use_coord:
            B, _, T, H, W = x.shape
            coord = generate_coord_tensor(B, T, H, W, x.device)
            x = torch.cat([x, coord], dim=1)

        feat = self.encoder(x)
        fusion = self.fusion_dilated(feat)
        feat = self.fusion_res(feat) + fusion

        B, C, T, H, W = feat.shape
        feat_reshape = feat.permute(2, 0, 3, 4, 1).reshape(T, B * H * W, C)
        for layer in self.temporal_transformer:
            feat_reshape = layer(feat_reshape)
        feat = feat_reshape.reshape(T, B, H, W, C).permute(1, 4, 0, 2, 3)

        feat = feat * self.attn_weight(feat)
        feat = feat + self.physics_gate(feat)

        feat = self.temporal_up(feat)
        out = self.spatial_up(feat)
        return out

    def summarize(self, input_shape=(6, 6, 5, 4), device='cuda'):
        """
        打印模型结构摘要。
        注意：传入原始输入通道数 c_in，其它尺寸自动推算。
        """
        dummy_input = torch.randn(1, self.c_in, *input_shape[1:]).to(device)
        summary(self.to(device), input_data=dummy_input, col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=3)





