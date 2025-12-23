import torch
from torch import nn
import torch.nn.functional as F


def generate_coord_tensor(B, T, H, W, device):
    """Generate normalized coordinate tensor: [B, 2, T, H, W]"""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W), indexing='ij')
    coords = torch.stack([xx, yy], dim=0)
    coords = coords.unsqueeze(1).repeat(1, T, 1, 1)
    coords = coords.unsqueeze(0).repeat(B, 1, 1, 1, 1)
    return coords.to(device)


class LSMGatedBlock(nn.Module):
    """Land-sea mask gating block for physics-guided feature modulation."""
    
    def __init__(self, in_channels, lsm_channels=1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(in_channels + lsm_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, lsm):
        lsm_upsampled = F.interpolate(lsm, size=x.shape[2:], mode='trilinear', align_corners=False)
        gating_input = torch.cat([x, lsm_upsampled], dim=1)
        gate_mask = self.gate(gating_input)
        return x * gate_mask


class STSRNetPlus(nn.Module):
    """Physics-guided spatiotemporal super-resolution network for wind fields."""
    
    def __init__(self, t_scale=6, s_scale=4, extra_scale=2.5, c_in=4, use_coord=True, use_lsm=True):
        super().__init__()
        self.t_scale = t_scale
        self.s_scale = s_scale
        self.extra_scale = extra_scale
        self.total_spatial_scale = self.s_scale * self.extra_scale
        self.use_coord = use_coord
        self.base_c_in = c_in
        self.c_in = c_in + 2 if use_coord else c_in
        self.c_in = c_in + 1 if use_lsm else c_in

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

        self.gate_block = LSMGatedBlock(in_channels=128)

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
        self.attn_maps = []
        if self.use_coord:
            B, _, T, H, W = x.shape
            coord = generate_coord_tensor(B, T, H, W, x.device)
            x = torch.cat([x, coord], dim=1)

        x_base, lsm = x[:, :-1], x[:, -1:]

        feat = self.encoder(x_base)
        fusion = self.fusion_dilated(feat)
        feat = self.fusion_res(feat) + fusion

        B, C, T, H, W = feat.shape
        feat_reshape = feat.permute(2, 0, 3, 4, 1).reshape(T, B * H * W, C)
        for layer in self.temporal_transformer:
            feat_reshape = layer(feat_reshape)
        feat = feat_reshape.reshape(T, B, H, W, C).permute(1, 4, 0, 2, 3)

        attn = self.attn_weight(feat)
        gate = self.physics_gate(feat)
        feat = feat * attn + gate

        feat = self.gate_block(feat, lsm)

        feat = self.temporal_up(feat)
        out = self.spatial_up(feat)
        return out


class STSRNetPlusCompare1(nn.Module):
    """Ablation variant without multi-scale fusion."""
    
    def __init__(self, t_scale=6, s_scale=4, extra_scale=2.5, c_in=4, use_coord=True, use_lsm=True):
        super().__init__()
        self.t_scale = t_scale
        self.s_scale = s_scale
        self.extra_scale = extra_scale
        self.total_spatial_scale = self.s_scale * self.extra_scale
        self.use_coord = use_coord
        self.c_in = c_in + 2 if use_coord else c_in
        self.c_in = c_in + 1 if use_lsm else c_in

        self.encoder = nn.Sequential(
            nn.Conv3d(self.c_in, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

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

        self.gate_block = LSMGatedBlock(in_channels=128)

        self.temporal_up = nn.Upsample(scale_factor=(t_scale, 1, 1), mode='trilinear', align_corners=False)

        self.spatial_up = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, self.total_spatial_scale, self.total_spatial_scale), mode='trilinear', align_corners=False),
            nn.Conv3d(64, 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        if self.use_coord:
            B, _, T, H, W = x.shape
            coord = generate_coord_tensor(B, T, H, W, x.device)
            x = torch.cat([x, coord], dim=1)

        x_base, lsm = x[:, :-1], x[:, -1:]

        feat = self.encoder(x_base)

        B, C, T, H, W = feat.shape
        feat_reshape = feat.permute(2, 0, 3, 4, 1).reshape(T, B * H * W, C)
        for layer in self.temporal_transformer:
            feat_reshape = layer(feat_reshape)
        feat = feat_reshape.reshape(T, B, H, W, C).permute(1, 4, 0, 2, 3)

        attn = self.attn_weight(feat)
        gate = self.physics_gate(feat)
        feat = feat * attn + gate

        feat = self.gate_block(feat, lsm)

        feat = self.temporal_up(feat)
        out = self.spatial_up(feat)
        return out


class STSRNetPlusCompare2(nn.Module):
    """Ablation variant without temporal transformer."""
    
    def __init__(self, t_scale=6, s_scale=4, extra_scale=2.5, c_in=4, use_coord=True, use_lsm=True):
        super().__init__()
        self.t_scale = t_scale
        self.s_scale = s_scale
        self.extra_scale = extra_scale
        self.total_spatial_scale = self.s_scale * self.extra_scale
        self.use_coord = use_coord
        self.c_in = c_in + 2 if use_coord else c_in
        self.c_in = c_in + 1 if use_lsm else c_in

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

        self.attn_weight = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )

        self.physics_gate = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv3d(128, 128, kernel_size=1)
        )

        self.gate_block = LSMGatedBlock(in_channels=128)

        self.temporal_up = nn.Upsample(scale_factor=(t_scale, 1, 1), mode='trilinear', align_corners=False)

        self.spatial_up = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, self.total_spatial_scale, self.total_spatial_scale), mode='trilinear', align_corners=False),
            nn.Conv3d(64, 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        if self.use_coord:
            B, _, T, H, W = x.shape
            coord = generate_coord_tensor(B, T, H, W, x.device)
            x = torch.cat([x, coord], dim=1)

        x_base, lsm = x[:, :-1], x[:, -1:]

        feat = self.encoder(x_base)
        fusion = self.fusion_dilated(feat)
        feat = self.fusion_res(feat) + fusion

        attn = self.attn_weight(feat)
        gate = self.physics_gate(feat)
        feat = feat * attn + gate

        feat = self.gate_block(feat, lsm)

        feat = self.temporal_up(feat)
        out = self.spatial_up(feat)
        return out


class STSRNetPlusCompare3(nn.Module):
    """Ablation variant without attention and physics gating."""
    
    def __init__(self, t_scale=6, s_scale=4, extra_scale=2.5, c_in=4, use_coord=True, use_lsm=True):
        super().__init__()
        self.t_scale = t_scale
        self.s_scale = s_scale
        self.extra_scale = extra_scale
        self.total_spatial_scale = self.s_scale * self.extra_scale
        self.use_coord = use_coord
        self.c_in = c_in + 2 if use_coord else c_in
        self.c_in = c_in + 1 if use_lsm else c_in

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

        self.gate_block = LSMGatedBlock(in_channels=128)

        self.temporal_up = nn.Upsample(scale_factor=(t_scale, 1, 1), mode='trilinear', align_corners=False)

        self.spatial_up = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, self.total_spatial_scale, self.total_spatial_scale), mode='trilinear', align_corners=False),
            nn.Conv3d(64, 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        if self.use_coord:
            B, _, T, H, W = x.shape
            coord = generate_coord_tensor(B, T, H, W, x.device)
            x = torch.cat([x, coord], dim=1)

        x_base, lsm = x[:, :-1], x[:, -1:]

        feat = self.encoder(x_base)
        fusion = self.fusion_dilated(feat)
        feat = self.fusion_res(feat) + fusion

        B, C, T, H, W = feat.shape
        feat_reshape = feat.permute(2, 0, 3, 4, 1).reshape(T, B * H * W, C)
        for layer in self.temporal_transformer:
            feat_reshape = layer(feat_reshape)
        feat = feat_reshape.reshape(T, B, H, W, C).permute(1, 4, 0, 2, 3)

        feat = self.gate_block(feat, lsm)

        feat = self.temporal_up(feat)
        out = self.spatial_up(feat)
        return out


class STSRNetPlusCompare4(nn.Module):
    """Ablation variant without LSM gating."""
    
    def __init__(self, t_scale=6, s_scale=4, extra_scale=2.5, c_in=4, use_coord=True, use_lsm=True):
        super().__init__()
        self.t_scale = t_scale
        self.s_scale = s_scale
        self.extra_scale = extra_scale
        self.total_spatial_scale = self.s_scale * self.extra_scale
        self.use_coord = use_coord
        self.c_in = c_in + 2 if use_coord else c_in
        self.c_in = c_in + 1 if use_lsm else c_in

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

        self.temporal_up = nn.Upsample(scale_factor=(t_scale, 1, 1), mode='trilinear', align_corners=False)

        self.spatial_up = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, self.total_spatial_scale, self.total_spatial_scale), mode='trilinear', align_corners=False),
            nn.Conv3d(64, 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        if self.use_coord:
            B, _, T, H, W = x.shape
            coord = generate_coord_tensor(B, T, H, W, x.device)
            x = torch.cat([x, coord], dim=1)

        x_base, _ = x[:, :-1], x[:, -1:]

        feat = self.encoder(x_base)
        fusion = self.fusion_dilated(feat)
        feat = self.fusion_res(feat) + fusion

        B, C, T, H, W = feat.shape
        feat_reshape = feat.permute(2, 0, 3, 4, 1).reshape(T, B * H * W, C)
        for layer in self.temporal_transformer:
            feat_reshape = layer(feat_reshape)
        feat = feat_reshape.reshape(T, B, H, W, C).permute(1, 4, 0, 2, 3)

        feat = self.temporal_up(feat)
        out = self.spatial_up(feat)
        return out

