import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import CottonDataset
from src.model import build_root_sam2
from src.utils import MetricCalculator, plot_curves, focal_dice_loss

CONFIG = {
    "ckpt": "checkpoints/sam2.1_hiera_large.pt",
    "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "data_root": "data/PRMI",
    "save_dir": "checkpoints/cotton_output",
    "batch_size": 4,           
    "lr": 1e-4,
    "epochs": 20,
    "eval_interval": 1,
    "num_workers": 8           
}

def run_epoch(model, loader, optimizer, device, is_train=True, desc=""):
    if is_train: model.train()
    else: model.eval()
    
    tracker = MetricCalculator()
    pbar = tqdm(loader, desc=desc, leave=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    for imgs, masks, boxes in pbar:
        imgs, masks, boxes = imgs.to(device), masks.to(device), boxes.to(device)
        box_prompts = boxes.unsqueeze(1) # [B, 1, 4]

        with torch.autocast(device_type="cuda", dtype=dtype):
            if is_train:
                optimizer.zero_grad()
                feat_dict = model.forward_image(imgs)
                sparse, dense = model.sam_prompt_encoder(
                    points=None, boxes=box_prompts, masks=None)
                
                low_res_masks, _, _, _ = model.sam_mask_decoder(
                    image_embeddings=feat_dict["image_embed"],
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=feat_dict["high_res_feats"]
                )
                
                # 插值回 768 计算 Loss
                pred = F.interpolate(low_res_masks, (768, 768), mode="bilinear", align_corners=False)
                loss = focal_dice_loss(pred, masks)
                
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    feat_dict = model.forward_image(imgs)
                    sparse, dense = model.sam_prompt_encoder(
                        points=None, boxes=box_prompts, masks=None)
                    low_res_masks, _, _, _ = model.sam_mask_decoder(
                        image_embeddings=feat_dict["image_embed"],
                        image_pe=model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse,
                        dense_prompt_embeddings=dense,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=feat_dict["high_res_feats"]
                    )
                    pred = F.interpolate(low_res_masks, (768, 768), mode="bilinear", align_corners=False)
                    loss = focal_dice_loss(pred, masks)
        
        tracker.update(pred.detach().float(), masks.float(), loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return tracker.compute()

def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running RootSAM2 (Encoder Only) on: {device}")
    
    # 1. Build Model
    model = build_root_sam2(CONFIG["ckpt"], CONFIG["model_cfg"]).to(device)
    
    # 2. Data
    train_ds = CottonDataset(CONFIG["data_root"], split='train')
    val_ds = CottonDataset(CONFIG["data_root"], split='val')
    test_ds = CottonDataset(CONFIG["data_root"], split='test')
    
    kw = {"batch_size": CONFIG["batch_size"], "num_workers": CONFIG["num_workers"], "pin_memory": True}
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    
    # 3. Optimizer (只优化 requires_grad=True 的参数)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_f1 = 0.0

    print("\n>>> Start Training...")
    for epoch in range(CONFIG["epochs"]):
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        t_stats = run_epoch(model, train_loader, optimizer, device, True, "Train")
        scheduler.step()
        
        if (epoch+1) % CONFIG["eval_interval"] == 0:
            v_stats = run_epoch(model, val_loader, optimizer, device, False, "Val")
            print(f"  [Val] Loss: {v_stats['loss']:.4f} | F1: {v_stats['f1']:.4f} | IoU: {v_stats['iou']:.4f}")
            
            history['train_loss'].append(t_stats['loss'])
            history['val_loss'].append(v_stats['loss'])
            history['val_f1'].append(v_stats['f1'])
            plot_curves(history, CONFIG["save_dir"])
            
            if v_stats['f1'] > best_f1:
                best_f1 = v_stats['f1']
                torch.save(model.state_dict(), f"{CONFIG['save_dir']}/best_model.pth")
                print("  >>> New Best F1! Model Saved.")

    # 4. Final Test
    print("\n>>> Evaluating Best Model on Test Set...")
    model.load_state_dict(torch.load(f"{CONFIG['save_dir']}/best_model.pth"))
    test_res = run_epoch(model, test_loader, None, device, False, "Test")
    print(f"========= FINAL RESULTS (RootSAM2 Encoder-Only) =========\n F1: {test_res['f1']:.4f} | IoU: {test_res['iou']:.4f}")
    with open(f"{CONFIG['save_dir']}/test_metrics.json", 'w') as f: json.dump(test_res, f)

if __name__ == "__main__":
    main()