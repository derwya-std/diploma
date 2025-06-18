from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class IMUNormalizer(nn.Module):
    def __init__(self, imu_dim: int = 6, momentum: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(imu_dim))
        self.register_buffer("running_var", torch.ones(imu_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor):
        if self.training:
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            self.running_mean = self.running_mean * (1 - self.momentum) + mean * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + var * self.momentum
            return (x - mean) / (var.sqrt() + self.eps)
        return (x - self.running_mean) / (self.running_var.sqrt() + self.eps)

class CBAM(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, c // r, False), nn.ReLU(True), nn.Linear(c // r, c, False))
        self.spconv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        ca = torch.sigmoid(self.mlp(x.mean((2, 3))) + self.mlp(x.view(b, c, -1).max(-1)[0])).view(b, c, 1, 1)
        x = x * ca
        sa = torch.sigmoid(self.spconv(torch.cat([x.mean(1, True), x.max(1, True)[0]], 1)))
        return x * sa

class ResCBAM(nn.Module):
    def __init__(self, cin, cout, stride=1, dil=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, dil, dilation=dil, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, dil, dilation=dil, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.cbam = CBAM(cout)
        self.relu = nn.ReLU(True)
        self.down = nn.Identity() if (stride == 1 and cin == cout) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idn = self.down(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(self.cbam(x) + idn)

class FrameEncoder(nn.Module):
    def __init__(self, in_ch: int, emb_dim: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.block1 = ResCBAM(32, 64, 2)
        self.block2 = ResCBAM(64, 128, 2)
        self.block3 = ResCBAM(128, 128, 1, dil=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)

class VisualDeltaEncoder(nn.Module):
    def __init__(self, emb_dim: int, mode: str = "sub"):
        super().__init__()
        self.mode = mode
        if mode == "cat":
            self.mlp = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.ReLU(True), nn.Linear(emb_dim, emb_dim))
        elif mode == "tcn":
            self.tcn = nn.Conv1d(emb_dim, emb_dim, 3, padding=2, dilation=2, groups=emb_dim)

    def forward(self, v: torch.Tensor):
        if self.mode == "sub":
            delta = v[:, 1:] - v[:, :-1]
        elif self.mode == "cat":
            delta = torch.cat([v[:, 1:], v[:, :-1]], -1)
            delta = self.mlp(delta)
        else:
            x = v.transpose(1, 2)
            delta = self.tcn(F.pad(x, (2, 0)))
            delta = delta.transpose(1, 2)[:, 1:]
        pad0 = torch.zeros_like(delta[:, :1])
        return torch.cat([pad0, delta], 1)

class IMUAttention(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.q = nn.Linear(d, h, False)
        self.k = nn.Linear(d, h, False)
        self.e = nn.Linear(h, 1, False)

    def forward(self, v, imu):
        a = self.e(torch.tanh(self.q(v).unsqueeze(1) + self.k(imu)))
        w = torch.softmax(a, 1)
        return (w * imu).sum(1)

class VIOCNN(nn.Module):
    def __init__(self, img_channels: int = 3, imu_dim: int = 6, emb_dim: int = 192, hidden_size: int = 256, gru_layers: int = 3, dropout_p: float = 0.2):
        super().__init__()
        self.frame_enc = FrameEncoder(img_channels, emb_dim)
        self.delta_enc = VisualDeltaEncoder(emb_dim, mode="sub")
        self.imu_norm = IMUNormalizer(imu_dim)
        self.imu_fc = nn.Sequential(nn.Linear(imu_dim, emb_dim), nn.ReLU(True), nn.Linear(emb_dim, emb_dim), nn.ReLU(True))
        self.imu_ln = nn.LayerNorm(emb_dim)
        self.attn = IMUAttention(emb_dim, hidden_size)
        self.drop = nn.Dropout(dropout_p)
        self.gru = nn.GRU(emb_dim, hidden_size, gru_layers, batch_first=True, bidirectional=True, dropout=0.25 if gru_layers > 1 else 0.0)
        self.pose_fc = nn.Linear(hidden_size, 6)
        self.conf_fc = nn.Linear(hidden_size, 1)

    def forward(self, images: torch.Tensor, imu: torch.Tensor, seq_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = images.shape
        _, _, M, _ = imu.shape
        v_feat = self.frame_enc(images.view(B * T, C, H, W)).view(B, T, -1)
        v_delta = self.delta_enc(v_feat)
        imu_emb = self.imu_fc(self.imu_norm(imu.view(B * T * M, -1)))
        imu_emb = self.imu_ln(imu_emb).view(B * T, M, -1)
        fused = self.attn(v_delta.view(B * T, -1), imu_emb).view(B, T, -1)
        fused = self.drop(fused)
        packed = nn.utils.rnn.pack_padded_sequence(fused, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        half = out.size(-1) // 2
        out = out[..., :half] + out[..., half:]
        poses = self.pose_fc(out)
        conf = torch.sigmoid(self.conf_fc(out))
        return poses, conf
