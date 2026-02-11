import torch
import torch.nn as nn
from sam2.build_sam import build_sam2
from .modules import LoRA_QKV, ParallelLWA

class LWA_Wrapper(nn.Module):
    def __init__(self, block, dim):
        super().__init__()
        self.block = block
        self.lwa = ParallelLWA(dim)
    
    def forward(self, x, *args, **kwargs):
        return self.block(x, *args, **kwargs) + self.lwa(x)

class SAM2_Root_Wrapper(nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
        # 方便外部调用
        self.image_encoder = sam_model.image_encoder
        self.sam_prompt_encoder = sam_model.sam_prompt_encoder
        self.sam_mask_decoder = sam_model.sam_mask_decoder

    def forward_image(self, img):
        backbone_out = self.image_encoder(img)
        
        if "backbone_fpn" in backbone_out:
            features = backbone_out["backbone_fpn"]
        elif "vision_features" in backbone_out and isinstance(backbone_out["vision_features"], list):
            features = backbone_out["vision_features"]
        else:
            raise KeyError(f"Unknown output: {backbone_out.keys()}")
        
        # [投影修正] 对高分辨率特征进行降维 (256->64/32)
        feat_s0 = features[0]
        feat_s1 = features[1]
        
        if hasattr(self.sam_mask_decoder, "conv_s0"):
            feat_s0 = self.sam_mask_decoder.conv_s0(feat_s0)
        if hasattr(self.sam_mask_decoder, "conv_s1"):
            feat_s1 = self.sam_mask_decoder.conv_s1(feat_s1)
            
        return {
            "image_embed": features[-1],
            "high_res_feats": [feat_s0, feat_s1]
        }

def build_root_sam2(ckpt_path, config_file):
    print(f"Building SAM 2 from {ckpt_path}...")
    sam_model = build_sam2(config_file, ckpt_path)
    
    # 1. 强制修改分辨率为 768 (解决 48 vs 64 报错)
    resolution = 768
    stride = 16
    feat_res = resolution // stride # 48
    print(f"Resizing PromptEncoder to {feat_res}x{feat_res} (Input: {resolution}x{resolution})")
    
    sam_model.image_size = resolution
    sam_model.sam_prompt_encoder.image_embedding_size = (feat_res, feat_res)
    sam_model.sam_prompt_encoder.input_image_size = (resolution, resolution)
    sam_model.sam_prompt_encoder.mask_input_size = (4 * feat_res, 4 * feat_res)

    # 2. 冻结全网 (作为基础)
    for p in sam_model.parameters(): p.requires_grad = False
    
    # 3. 注入 LoRA (可训练)
    print("Injecting LoRA...")
    for name, module in sam_model.image_encoder.named_modules():
        if "qkv" in name and isinstance(module, nn.Linear):
            parent = sam_model.image_encoder.get_submodule(name.rsplit('.', 1)[0])
            child = name.rsplit('.', 1)[1]
            lora_layer = LoRA_QKV(module, rank=16, alpha=16)
            setattr(parent, child, lora_layer)
            for p in lora_layer.parameters(): p.requires_grad = True

    # 4. 注入 LWA (可训练，跳过变维层)
    backbone = sam_model.image_encoder.trunk if hasattr(sam_model.image_encoder, "trunk") else sam_model.image_encoder
    print("Injecting LWA...")
    injected_count = 0
    for i in range(8):
        if i >= len(backbone.blocks): break
        block = backbone.blocks[i]
        # 跳过 Transition Block
        if hasattr(block, 'dim') and hasattr(block, 'dim_out') and block.dim != block.dim_out:
            print(f"  Skipping Block {i}: Transition ({block.dim}->{block.dim_out})")
            continue
        
        backbone.blocks[i] = LWA_Wrapper(block, block.dim)
        injected_count += 1
        for p in backbone.blocks[i].lwa.parameters(): p.requires_grad = True
    
    print(f"Successfully injected {injected_count} LWA modules.")

    # 5. [核心设定] 保持 Decoder 和 Prompt Encoder 冻结
    # 仅训练 Image Encoder 里的 LoRA 和 LWA 旁路
    for p in sam_model.sam_mask_decoder.parameters(): p.requires_grad = True
    for p in sam_model.sam_prompt_encoder.parameters(): p.requires_grad = False
    
    # 打印参数统计
    trainable = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in sam_model.parameters())
    print(f"\n[Model Config] RootSAM2 (Encoder Adapters Only)")
    print(f"  Trainable Params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total*100:.2f}%)")
    
    return SAM2_Root_Wrapper(sam_model)