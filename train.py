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

# === CONFIG ===
# 你显存充足：建议 1024 起步。若还想更细，可尝试 1280/1536（注意显存与吞吐）。
CONFIG = {
    "ckpt": "checkpoints/sam2.1_hiera_large.pt",
    "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "data_root": "data/PRMI",
    "save_dir": "checkpoints/cotton_output_diff_lr",
    "img_size": 1024,
    "batch_size": 2,
    "lr_encoder": 1e-4,  # LoRA/LWA 的大学习率
    "lr_decoder": 1e-6,  # Decoder 的微小学习率 (防止遗忘)
    "epochs": 20,
    "eval_interval": 1,
    "num_workers": 4,
    # 可选：用第一次预测 mask 作为 prompt 再 refine 一次（对细根通常有帮助）
    "use_mask_refine": True,
    # 训练集 zoom-crop 概率（crop 尺寸在 dataset 内部多尺度随机）
    "crop_prob": 0.6,
}


def _forward_once(model, feat_dict, box_prompts, mask_prompt=None):
    """一次 SAM 头部推理：boxes + (optional) masks prompt -> low_res_masks"""
    sparse, dense = model.sam_prompt_encoder(points=None, boxes=box_prompts, masks=mask_prompt)
    low_res_masks, _, _, _ = model.sam_mask_decoder(
        image_embeddings=feat_dict["image_embed"],
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=feat_dict["high_res_feats"],
    )
    return low_res_masks


def run_epoch(model, loader, optimizer, device, is_train=True, desc="", use_mask_refine=False):
    if is_train and optimizer is None:
        raise ValueError("optimizer cannot be None when is_train=True")

    model.train() if is_train else model.eval()

    tracker = MetricCalculator()
    pbar = tqdm(loader, desc=desc, leave=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for imgs, masks, boxes in pbar:
            imgs, masks, boxes = imgs.to(device), masks.to(device), boxes.to(device)
            box_prompts = boxes.unsqueeze(1)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=dtype):
                feat_dict = model.forward_image(imgs)

                # Pass 1: boxes only
                low_res_masks = _forward_once(model, feat_dict, box_prompts, mask_prompt=None)

                # Optional Pass 2: mask prompt refine
                if use_mask_refine:
                    mask_in_size = model.sam_prompt_encoder.mask_input_size
                    mask_prompt = torch.sigmoid(low_res_masks)
                    mask_prompt = F.interpolate(mask_prompt, size=mask_in_size, mode="bilinear", align_corners=False)
                    mask_prompt = mask_prompt.detach()
                    low_res_masks = _forward_once(model, feat_dict, box_prompts, mask_prompt=mask_prompt)

                # ✅ 不写死 768；直接对齐 GT 尺寸
                pred = F.interpolate(low_res_masks, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = focal_dice_loss(pred, masks)

            if is_train:
                loss.backward()
                optimizer.step()

            tracker.update(pred.detach().float(), masks.float(), float(loss.item()))
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return tracker.compute()


def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running RootSAM2 (Diff LR) on: {device}")

    # 1) Build Model (resolution 与 dataset 对齐)
    model = build_root_sam2(CONFIG["ckpt"], CONFIG["model_cfg"], resolution=CONFIG["img_size"]).to(device)

    # 2) Data
    # 训练：允许 upsample（放大镜）+ 多尺度 zoom-crop
    train_ds = CottonDataset(
        CONFIG["data_root"],
        split="train",
        img_size=CONFIG["img_size"],
        keep_ratio=True,
        crop_prob=CONFIG["crop_prob"],
        allow_upscale=True,
    )
    # 验证/测试：更保真——不放大原图，只 padding 到 img_size
    val_ds = CottonDataset(
        CONFIG["data_root"],
        split="val",
        img_size=CONFIG["img_size"],
        keep_ratio=True,
        crop_prob=0.0,
        allow_upscale=False,
    )
    test_ds = CottonDataset(
        CONFIG["data_root"],
        split="test",
        img_size=CONFIG["img_size"],
        keep_ratio=True,
        crop_prob=0.0,
        allow_upscale=False,
    )

    kw = {"batch_size": CONFIG["batch_size"], "num_workers": CONFIG["num_workers"], "pin_memory": True}
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    # 3) Differential Learning Rates
    encoder_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
    decoder_params = list(model.sam_mask_decoder.parameters()) + list(model.sam_prompt_encoder.parameters())
    decoder_params = [p for p in decoder_params if p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": encoder_params, "lr": CONFIG["lr_encoder"]},
            {"params": decoder_params, "lr": CONFIG["lr_decoder"]},
        ]
    )

    print("Optimizer Config:")
    print(f"  - Encoder Adapters LR: {CONFIG['lr_encoder']} (Params: {len(encoder_params)})")
    print(f"  - Decoder Heads LR:    {CONFIG['lr_decoder']} (Params: {len(decoder_params)})")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = 0.0

    print("\n>>> Start Training...")
    for epoch in range(CONFIG["epochs"]):
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}")
        t_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            is_train=True,
            desc="Train",
            use_mask_refine=CONFIG["use_mask_refine"],
        )
        scheduler.step()

        if (epoch + 1) % CONFIG["eval_interval"] == 0:
            v_stats = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                is_train=False,
                desc="Val",
                use_mask_refine=CONFIG["use_mask_refine"],
            )
            print(f"  [Val] Loss: {v_stats['loss']:.4f} | F1: {v_stats['f1']:.4f} | IoU: {v_stats['iou']:.4f}")

            history["train_loss"].append(t_stats["loss"])
            history["val_loss"].append(v_stats["loss"])
            history["val_f1"].append(v_stats["f1"])
            plot_curves(history, CONFIG["save_dir"])

            if v_stats["f1"] > best_f1:
                best_f1 = v_stats["f1"]
                torch.save(model.state_dict(), f"{CONFIG['save_dir']}/best_model.pth")
                print("  >>> New Best F1! Model Saved.")

    # Final Test
    print("\n>>> Evaluating Best Model on Test Set...")
    model.load_state_dict(torch.load(f"{CONFIG['save_dir']}/best_model.pth", map_location=device))
    test_res = run_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        is_train=False,
        desc="Test",
        use_mask_refine=CONFIG["use_mask_refine"],
    )
    print(f"========= FINAL RESULTS =========\n F1: {test_res['f1']:.4f} | IoU: {test_res['iou']:.4f}")
    with open(f"{CONFIG['save_dir']}/test_metrics.json", "w") as f:
        json.dump(test_res, f, indent=2)


if __name__ == "__main__":
    main()
