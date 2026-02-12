import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class CottonDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 1024,
        keep_ratio: bool = True,
        crop_prob: float = 0.6,
        crop_size: int = 512,
        pad_value: int = 0,
        allow_upscale: bool = True,
        crop_choices=(256, 384, 512),
    ):
        """RootSAM2 数据集加载器（保真 + 细根友好）

        关键点：
        1) keep_ratio=True 时：等比 resize + padding 到 img_size（避免非等比拉伸带来的细根变形）
        2) 训练集：zoom-crop + 放大（allow_upscale=True），并且 crop 多尺度（256/384/512）
        3) val/test：默认不放大原图（allow_upscale=False），只 padding（更保真、更稳定）
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.keep_ratio = keep_ratio
        self.crop_prob = crop_prob if split == "train" else 0.0
        self.crop_size = crop_size
        self.pad_value = pad_value
        self.allow_upscale = allow_upscale
        self.crop_choices = tuple(crop_choices)

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
            mask_path = os.path.join(mask_dir, f"GT_{name_no_ext}.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, f"{name_no_ext}.png")

            if os.path.exists(mask_path):
                valid_samples.append((img_path, mask_path))

        print(f"[{self.split.upper()}] Loaded: {len(valid_samples)} pairs from {self.target_subset}")
        return valid_samples

    def __len__(self):
        return len(self.samples)

    def _maybe_zoom_crop(self, image: np.ndarray, mask: np.ndarray):
        """训练集：从根系区域附近裁小块，再放大到 img_size（多尺度放大细侧根）"""
        if self.crop_prob <= 0.0 or np.random.rand() >= self.crop_prob:
            return image, mask

        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return image, mask

        h, w = mask.shape
        # 多尺度：更利于极细侧根
        cs = int(np.random.choice(self.crop_choices))
        cs = int(min(cs, h, w))
        if cs < 16:
            return image, mask

        cy = int(np.random.choice(ys))
        cx = int(np.random.choice(xs))

        y1 = max(0, cy - cs // 2)
        x1 = max(0, cx - cs // 2)
        y2 = min(h, y1 + cs)
        x2 = min(w, x1 + cs)
        y1 = max(0, y2 - cs)
        x1 = max(0, x2 - cs)

        return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def _resize_and_pad(self, image: np.ndarray, mask: np.ndarray):
        """等比缩放到长边 <= img_size，然后 padding 到 img_size×img_size

        allow_upscale=True ：可放大（训练 crop 场景推荐）
        allow_upscale=False：不放大原图，只 padding（val/test 推荐）
        """
        h, w = image.shape[:2]

        if self.allow_upscale:
            scale = self.img_size / max(h, w)
        else:
            scale = min(1.0, self.img_size / max(h, w))

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        # upsample 用更锐的插值，downsample 用 area 更稳
        interp_img = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA

        image_r = cv2.resize(image, (new_w, new_h), interpolation=interp_img)
        mask_r = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        pad_w = self.img_size - new_w
        pad_h = self.img_size - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        image_p = cv2.copyMakeBorder(
            image_r,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(self.pad_value, self.pad_value, self.pad_value),
        )
        mask_p = cv2.copyMakeBorder(
            mask_r,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        return image_p, mask_p

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # 1) Load
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        # 2) Zoom-crop (train only)
        image, mask = self._maybe_zoom_crop(image, mask)

        # 3) Resize & Pad (keep_ratio=True) OR naive resize
        if self.keep_ratio:
            image, mask = self._resize_and_pad(image, mask)
        else:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 4) Normalize & Float32
        image = image.astype(np.float32) / 255.0
        image = (image - self.pixel_mean) / self.pixel_std

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_bin = (mask > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0).float()

        # 5) Prompt box（在最终尺寸上算，坐标与输入一致）
        y_idxs, x_idxs = np.where(mask_bin > 0)
        if len(y_idxs) > 0:
            x_min, x_max = int(np.min(x_idxs)), int(np.max(x_idxs))
            y_min, y_max = int(np.min(y_idxs)), int(np.max(y_idxs))

            # noise 随分辨率缩放
            max_noise = max(5, self.img_size // 40)
            noise = int(np.random.randint(0, max_noise + 1))

            box = np.array(
                [
                    max(0, x_min - noise),
                    max(0, y_min - noise),
                    min(self.img_size - 1, x_max + noise),
                    min(self.img_size - 1, y_max + noise),
                ],
                dtype=np.float32,
            )
        else:
            box = np.array([0, 0, self.img_size - 1, self.img_size - 1], dtype=np.float32)

        return image_tensor, mask_tensor, torch.from_numpy(box).float()
