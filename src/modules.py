import torch
import torch.nn as nn
import math

# --- 1. Parallel LWA (Haar Wavelet) ---
def haar_dwt(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    LL = x1 + x2 + x3 + x4
    HL = -x1 - x2 + x3 + x4
    LH = -x1 + x2 - x3 + x4
    HH = x1 - x2 - x3 + x4
    return LL, torch.cat((HL, LH, HH), 1)

def haar_idwt(LL, high):
    HL, LH, HH = torch.chunk(high, 3, dim=1)
    x1 = (LL - HL - LH + HH) / 2
    x2 = (LL - HL + LH - HH) / 2
    x3 = (LL + HL - LH - HH) / 2
    x4 = (LL + HL + LH + HH) / 2
    B, C, H, W = LL.shape
    out = torch.zeros((B, C, H*2, W*2), device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, 0::2] = x1
    out[:, :, 1::2, 0::2] = x2
    out[:, :, 0::2, 1::2] = x3
    out[:, :, 1::2, 1::2] = x4
    return out

class ParallelLWA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim*3, dim*3, 1, groups=dim),
            nn.Sigmoid()
        )
        self.zero_conv = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x_in = x.permute(0, 3, 1, 2).contiguous()
        LL, high = haar_dwt(x_in)
        high_enhanced = high * self.gate(high)
        out = haar_idwt(LL, high_enhanced)
        out = self.zero_conv(out)
        return out.permute(0, 2, 3, 1).contiguous() * self.scale

# --- 2. LoRA for QKV (Auto-Dimension) ---
class LoRA_QKV(nn.Module):
    def __init__(self, original_layer, rank=16, alpha=16, dropout=0.05):
        super().__init__()
        self.qkv = original_layer
        
        # 自动推断维度
        self.in_dim = original_layer.in_features
        self.out_dim = original_layer.out_features // 3 # Q,K,V 分离
        
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        
        # 映射到 out_dim
        self.lora_A_q = nn.Linear(self.in_dim, rank, bias=False)
        self.lora_B_q = nn.Linear(rank, self.out_dim, bias=False)
        
        self.lora_A_v = nn.Linear(self.in_dim, rank, bias=False)
        self.lora_B_v = nn.Linear(rank, self.out_dim, bias=False)
        
        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_v.weight)
        
        for p in self.qkv.parameters(): p.requires_grad = False

    def forward(self, x):
        base_out = self.qkv(x)
        dx = self.dropout(x)
        dq = self.lora_B_q(self.lora_A_q(dx)) * self.scaling
        dv = self.lora_B_v(self.lora_A_v(dx)) * self.scaling
        
        q, k, v = base_out.chunk(3, dim=-1)
        return torch.cat([q + dq, k, v + dv], dim=-1)