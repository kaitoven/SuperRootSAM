import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# Upgrade A: 可学习的小波变换 (Learnable Haar, 正值系数保证稳定)
# ==========================================
class LearnableHaarDWT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 用 raw 参数 + softplus 保证为正，避免出现符号翻转导致的偶发崩溃
        # softplus(0)=0.693；如希望更接近 0.5，可把初值设为 -0.45 左右
        self.raw_scale = nn.Parameter(torch.tensor(-0.45))
        self.raw_diff = nn.Parameter(torch.tensor(-0.45))
        self.eps = 1e-6

    def _scale(self):
        return F.softplus(self.raw_scale) + self.eps

    def _diff(self):
        return F.softplus(self.raw_diff) + self.eps

    def forward(self, x):
        # x: [B, C, H, W]
        s = self._scale()
        d = self._diff()

        x01 = x[:, :, 0::2, :]
        x02 = x[:, :, 1::2, :]
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        # LL: 低频
        LL = (x1 + x2 + x3 + x4) * s

        # 高频
        HL = (-x1 - x2 + x3 + x4) * d
        LH = (-x1 + x2 - x3 + x4) * d
        HH = (x1 - x2 - x3 + x4) * d

        return LL, torch.cat((HL, LH, HH), 1)


class LearnableHaarIDWT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, LL, high, scale, diff_scale):
        """与 LearnableHaarDWT 配对的可逆 IDWT

        DWT 定义：
          LL = (x1+x2+x3+x4)*scale
          HL = (-x1-x2+x3+x4)*diff
          LH = (-x1+x2-x3+x4)*diff
          HH = (x1-x2-x3+x4)*diff

        令：
          A = LL/scale
          B = HL/diff, C = LH/diff, D = HH/diff
        则：
          x1 = (A - B - C + D)/4, ...（其余同理）
        """
        HL, LH, HH = torch.chunk(high, 3, dim=1)

        A = LL / scale
        B = HL / diff_scale
        C = LH / diff_scale
        D = HH / diff_scale

        x1 = (A - B - C + D) / 4.0
        x2 = (A - B + C - D) / 4.0
        x3 = (A + B - C - D) / 4.0
        x4 = (A + B + C + D) / 4.0

        Bsz, Cch, H, W = LL.shape
        out = torch.zeros((Bsz, Cch, H * 2, W * 2), device=LL.device, dtype=LL.dtype)
        out[:, :, 0::2, 0::2] = x1
        out[:, :, 1::2, 0::2] = x2
        out[:, :, 0::2, 1::2] = x3
        out[:, :, 1::2, 1::2] = x4
        return out


# ==========================================
# Upgrade B: 双重门控 (Spatial + Channel)
# ==========================================
class DualGate(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # Channel Attention
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 3, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim * 3, 1),
            nn.Sigmoid(),
        )

        # Spatial Attention (7x7)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim * 3, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

        self.fusion = nn.Conv2d(dim * 3, dim * 3, 1)

    def forward(self, x):
        # x: [B, 3C, H, W]
        c_att = self.channel_gate(x)
        x_c = x * c_att

        s_att = self.spatial_gate(x)
        x_s = x * s_att

        return self.fusion(x_c + x_s)


# ==========================================
# 主模块: Parallel LWA
# ==========================================
class ParallelLWA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dwt = LearnableHaarDWT(dim)
        self.idwt = LearnableHaarIDWT()

        self.gate = DualGate(dim)

        self.zero_conv = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # Input: [B, H, W, C] -> [B, C, H, W]
        x_in = x.permute(0, 3, 1, 2).contiguous()

        LL, high = self.dwt(x_in)
        high_enhanced = self.gate(high)

        # ✅ 使用 dwt 的正值系数，保证 DWT/IDWT 匹配与稳定
        out = self.idwt(LL, high_enhanced, self.dwt._scale(), self.dwt._diff())
        out = self.zero_conv(out)

        return out.permute(0, 2, 3, 1).contiguous() * self.scale


# ==========================================
# LoRA
# ==========================================
class LoRA_QKV(nn.Module):
    def __init__(self, original_layer, rank=32, alpha=16, dropout=0.05):
        super().__init__()
        self.qkv = original_layer
        self.in_dim = original_layer.in_features
        self.out_dim = original_layer.out_features // 3

        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        self.lora_A_q = nn.Linear(self.in_dim, rank, bias=False)
        self.lora_B_q = nn.Linear(rank, self.out_dim, bias=False)
        self.lora_A_v = nn.Linear(self.in_dim, rank, bias=False)
        self.lora_B_v = nn.Linear(rank, self.out_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_v.weight)

        for p in self.qkv.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.qkv(x)

        dx = self.dropout(x)
        dq = self.lora_B_q(self.lora_A_q(dx)) * self.scaling
        dv = self.lora_B_v(self.lora_A_v(dx)) * self.scaling

        q, k, v = base_out.chunk(3, dim=-1)
        return torch.cat([q + dq, k, v + dv], dim=-1)
