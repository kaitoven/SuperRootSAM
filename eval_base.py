import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sam2.build_sam import build_sam2
from src.dataset import CottonDataset
from src.utils import MetricCalculator
from src.model import SAM2_Root_Wrapper

# === 配置 ===
CONFIG = {
    "ckpt": "checkpoints/sam2.1_hiera_large.pt",       # 官方权重
    "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml", # 官方配置
    "data_root": "data/PRMI",
    "batch_size": 4,  # 评估时显存占用较小，4是安全的
    "img_size": 768   # 必须与 Fine-tuning 时保持一致
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Baseline Evaluation on: {device}")

    # 1. 加载官方 Vanilla SAM 2.1 (无 LoRA, 无 LWA)
    print("Loading Base SAM 2.1 (Zero-shot)...")
    model = build_sam2(CONFIG["model_cfg"], CONFIG["ckpt"])

    # 2. [关键] 强制修改分辨率为 768 (适配 CottonDataset)
    # 如果不加这一步，官方模型默认是 1024，会报错
    resolution = CONFIG["img_size"]
    stride = 16
    feat_res = resolution // stride
    print(f"Adjusting resolution to {resolution}x{resolution} (feat: {feat_res}x{feat_res})...")
    
    model.image_size = resolution
    model.sam_prompt_encoder.image_embedding_size = (feat_res, feat_res)
    model.sam_prompt_encoder.input_image_size = (resolution, resolution)
    model.sam_prompt_encoder.mask_input_size = (4 * feat_res, 4 * feat_res)

    # 3. 使用 Wrapper 包装 (复用其中的特征投影逻辑，但不包含 LoRA/LWA)
    # 注意：我们传入的是原始 model，没有经过 build_root_sam2 的注入过程
    model = SAM2_Root_Wrapper(model).to(device)
    model.eval()

    # 4. 加载测试集
    test_ds = CottonDataset(CONFIG["data_root"], split='test', img_size=CONFIG["img_size"])
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
    
    # 5. 开始评估
    tracker = MetricCalculator()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    print(f"Evaluating on {len(test_ds)} test images...")
    
    with torch.no_grad():
        for imgs, masks, boxes in tqdm(test_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            boxes = boxes.to(device)
            box_prompts = boxes.unsqueeze(1) # [B, 1, 4]

            # 开启混合精度加速
            with torch.autocast(device_type="cuda", dtype=dtype):
                # Forward Image (Wrapper 会处理 high_res_features 的投影)
                feat_dict = model.forward_image(imgs)
                
                # Prompt Encoder
                sparse, dense = model.sam_prompt_encoder(
                    points=None, boxes=box_prompts, masks=None)
                
                # Mask Decoder
                low_res_masks, _, _, _ = model.sam_mask_decoder(
                    image_embeddings=feat_dict["image_embed"],
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=feat_dict["high_res_feats"]
                )
                
                # 插值回 768x768
                pred = F.interpolate(low_res_masks, (resolution, resolution), 
                                   mode="bilinear", align_corners=False)
            
            # 更新指标
            tracker.update(pred.detach().float(), masks.float())

    # 6. 输出结果
    res = tracker.compute()
    print("\n" + "="*40)
    print(f" BASELINE SAM 2.1 (Large) RESULTS")
    print("="*40)
    print(f" F1 Score  : {res['f1']:.4f}")
    print(f" IoU       : {res['iou']:.4f}")
    print(f" Accuracy  : {res['acc']:.4f}")
    print(f" Precision : {res['prec']:.4f}")
    print(f" Recall    : {res['rec']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()