import os
import json
import torch
import torch.nn.functional as F

# ✅ 关键：无显示环境强制用 Agg，保证能落盘保存 png
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def focal_dice_loss(pred_logits, gt_mask):
    # Focal Loss
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask, reduction='none')
    pt = torch.exp(-bce)
    focal_loss = (0.5 * (1 - pt) ** 2 * bce).mean()

    # Dice Loss
    pred_prob = torch.sigmoid(pred_logits)
    smooth = 1e-5
    intersection = (pred_prob * gt_mask).sum(dim=(2, 3))
    union = pred_prob.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3))
    dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

    return focal_loss + dice_loss.mean()


class MetricCalculator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0
        self.count, self.total_loss = 0, 0

    def update(self, pred_logits, gt_mask, loss_val=0):
        pred_mask = (pred_logits > 0.0).float()
        self.tp += (pred_mask * gt_mask).sum().item()
        self.fp += (pred_mask * (1 - gt_mask)).sum().item()
        self.tn += ((1 - pred_mask) * (1 - gt_mask)).sum().item()
        self.fn += ((1 - pred_mask) * gt_mask).sum().item()
        self.total_loss += float(loss_val)
        self.count += 1

    def compute(self):
        smooth = 1e-6
        prec = self.tp / (self.tp + self.fp + smooth)
        rec = self.tp / (self.tp + self.fn + smooth)
        f1 = 2 * (prec * rec) / (prec + rec + smooth)
        iou = self.tp / (self.tp + self.fp + self.fn + smooth)
        acc = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + smooth)
        return {
            "loss": self.total_loss / (self.count + smooth),
            "f1": f1,
            "iou": iou,
            "acc": acc,
            "prec": prec,
            "rec": rec
        }


def _safe_savefig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_curves(history, save_dir):
    """
    history 期望包含：
      - history['train_loss']: list[float]
      - history['val_loss']:   list[float]
      - history['val_f1']:     list[float]
    """
    os.makedirs(save_dir, exist_ok=True)

    # ✅ 同步把 history 落盘，方便你检查是否确实有记录
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # 若还没记录到任何值，直接返回（避免空图）
    if len(history.get("val_f1", [])) == 0:
        return

    # 默认：train_loss/val_loss/val_f1 每次 eval 追加一次，所以长度一致
    n = len(history["val_f1"])
    epochs = list(range(1, n + 1))

    # 1) Train/Val Loss 同图（更直观）
    if len(history.get("train_loss", [])) == n and len(history.get("val_loss", [])) == n:
        plt.figure()
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.plot(epochs, history["val_loss"], label="Val Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        _safe_savefig(os.path.join(save_dir, "curve_losses.png"))

        # 2) 单独保存 Train Loss
        plt.figure()
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.title("Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        _safe_savefig(os.path.join(save_dir, "curve_train_loss.png"))

        # 3) 单独保存 Val Loss
        plt.figure()
        plt.plot(epochs, history["val_loss"], label="Val Loss")
        plt.title("Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        _safe_savefig(os.path.join(save_dir, "curve_val_loss.png"))

    # 4) Val F1
    plt.figure()
    plt.plot(epochs, history["val_f1"], label="Val F1")
    plt.title(f"Validation F1 (Best: {max(history['val_f1']):.4f})")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    _safe_savefig(os.path.join(save_dir, "curve_val_f1.png"))
