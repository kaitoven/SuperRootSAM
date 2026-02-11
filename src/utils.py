import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import json

def focal_dice_loss(pred_logits, gt_mask):
    # Focal Loss
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask, reduction='none')
    pt = torch.exp(-bce)
    focal_loss = (0.5 * (1 - pt)**2 * bce).mean() 
    
    # Dice Loss
    pred_prob = torch.sigmoid(pred_logits)
    smooth = 1e-5
    intersection = (pred_prob * gt_mask).sum(dim=(2,3))
    union = pred_prob.sum(dim=(2,3)) + gt_mask.sum(dim=(2,3))
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
        self.total_loss += loss_val
        self.count += 1
        
    def compute(self):
        smooth = 1e-6
        prec = self.tp / (self.tp + self.fp + smooth)
        rec = self.tp / (self.tp + self.fn + smooth)
        f1 = 2 * (prec * rec) / (prec + rec + smooth)
        iou = self.tp / (self.tp + self.fp + self.fn + smooth)
        acc = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + smooth)
        return {"loss": self.total_loss/(self.count+smooth), "f1": f1, "iou": iou, "acc": acc, "prec": prec, "rec": rec}

def plot_curves(history, save_dir):
    epochs = range(1, len(history['val_f1']) + 1)
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(epochs, history['val_f1'], label='Val F1', color='green')
    plt.title(f"Validation F1 (Best: {max(history['val_f1']):.4f})")
    plt.xlabel('Epoch'); plt.ylabel('F1 Score')
    plt.savefig(os.path.join(save_dir, "curve_val_f1.png"))
    plt.close()