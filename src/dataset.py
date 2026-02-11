import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CottonDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=768):
        """
        RootSAM2 专用数据集加载器
        - Resolution: 768x768 (适配 Patch Size=32，且显存友好)
        - Dtype: Float32 (避免 Double 冲突)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.target_subset = "Cotton_736x552_DPI150"
        
        # ImageNet 归一化参数 (Float32)
        self.pixel_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.pixel_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        self.samples = self._scan_dataset()

    def _scan_dataset(self):
        valid_samples = []
        img_dir = os.path.join(self.root_dir, self.split, "images", self.target_subset)
        mask_dir = os.path.join(self.root_dir, self.split, "masks_pixel_gt", self.target_subset)
        
        if not os.path.exists(img_dir):
            print(f"[Warning] Path not found: {img_dir}")
            return []

        img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        
        for img_path in img_paths:
            name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            # 兼容 GT_ 前缀
            mask_path = os.path.join(mask_dir, f"GT_{name_no_ext}.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, f"{name_no_ext}.png")
            
            if os.path.exists(mask_path):
                valid_samples.append((img_path, mask_path))
        
        print(f"[{self.split.upper()}] Loaded: {len(valid_samples)} pairs from {self.target_subset}")
        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # 1. 读取
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        
        # 2. Resize (统一拉伸到 768x768，不填充黑边以节省计算)
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # 3. Normalize & Float32
        image = image.astype(np.float32) / 255.0
        image = (image - self.pixel_mean) / self.pixel_std
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = (mask > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        # 4. Prompt 生成 (带随机噪声)
        y_idxs, x_idxs = np.where(mask > 0)
        if len(y_idxs) > 0:
            x_min, x_max = np.min(x_idxs), np.max(x_idxs)
            y_min, y_max = np.min(y_idxs), np.max(y_idxs)
            
            noise = np.random.randint(0, 20) # 模拟用户误差
            box = np.array([
                max(0, x_min - noise), max(0, y_min - noise),
                min(self.img_size, x_max + noise), min(self.img_size, y_max + noise)
            ])
        else:
            box = np.array([0, 0, self.img_size, self.img_size])

        return image_tensor, mask_tensor, torch.from_numpy(box).float()