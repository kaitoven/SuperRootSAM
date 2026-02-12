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
        # 兼容 Hiera Block 的参数传递
        return self.block(x, *args, **kwargs) + self.lwa(x)


class SAM2_Root_Wrapper(nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
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

        # 对高分辨率特征进行降维 (256->64/32)
        feat_s0 = features[0]
        feat_s1 = features[1]

        if hasattr(self.sam_mask_decoder, "conv_s0"):
            feat_s0 = self.sam_mask_decoder.conv_s0(feat_s0)
        if hasattr(self.sam_mask_decoder, "conv_s1"):
            feat_s1 = self.sam_mask_decoder.conv_s1(feat_s1)

        return {"image_embed": features[-1], "high_res_feats": [feat_s0, feat_s1]}


def build_root_sam2(ckpt_path, config_file, resolution: int = 1024):
    print(f"Building SAM 2 from {ckpt_path}...")
    sam_model = build_sam2(config_file, ckpt_path)

    # 1) 根据分辨率修改 PromptEncoder 的网格尺寸
    stride = 16
    feat_res = resolution // stride
    print(f"Resizing PromptEncoder to {feat_res}x{feat_res} (Input: {resolution}x{resolution})")

    sam_model.image_size = resolution
    sam_model.sam_prompt_encoder.image_embedding_size = (feat_res, feat_res)
    sam_model.sam_prompt_encoder.input_image_size = (resolution, resolution)
    sam_model.sam_prompt_encoder.mask_input_size = (4 * feat_res, 4 * feat_res)

    # 2) 冻结全网参数
    for p in sam_model.parameters():
        p.requires_grad = False

    # 3) 注入 LoRA (Rank=32)
    print("Injecting LoRA (Rank=32)...")
    for name, module in sam_model.image_encoder.named_modules():
        if "qkv" in name and isinstance(module, nn.Linear):
            parent = sam_model.image_encoder.get_submodule(name.rsplit(".", 1)[0])
            child = name.rsplit(".", 1)[1]
            lora_layer = LoRA_QKV(module, rank=32, alpha=16)
            setattr(parent, child, lora_layer)
            for p in lora_layer.parameters():
                p.requires_grad = True

    # 4) 注入 LWA（遍历所有 blocks，自动跳过变维层）
    image_encoder = sam_model.image_encoder
    backbone = image_encoder.trunk if hasattr(image_encoder, "trunk") else image_encoder

    print("Injecting LWA (All Stages)...")
    injected_count = 0
    for i in range(len(backbone.blocks)):
        block = backbone.blocks[i]

        if hasattr(block, "dim") and hasattr(block, "dim_out") and block.dim != block.dim_out:
            print(f"  Skipping Block {i} (Stage change): Transition ({block.dim}->{block.dim_out})")
            continue

        backbone.blocks[i] = LWA_Wrapper(block, block.dim)
        injected_count += 1
        for p in backbone.blocks[i].lwa.parameters():
            p.requires_grad = True

    print(f"Successfully injected {injected_count} LWA modules across ALL stages.")

    # 5) 解冻 Mask Decoder（Prompt Encoder 仍保持冻结）
    print("Unfreezing Mask Decoder...")
    for p in sam_model.sam_mask_decoder.parameters():
        p.requires_grad = True
    for p in sam_model.sam_prompt_encoder.parameters():
        p.requires_grad = False

    return SAM2_Root_Wrapper(sam_model)
