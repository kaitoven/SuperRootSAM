import os
import glob

# 配置你的路径
data_root = "data/PRMI" 
target_subset = "Cotton_736x552_DPI150"

print(f"Checking data root: {os.path.abspath(data_root)}")

for split in ['train', 'val', 'test']:
    img_dir = os.path.join(data_root, split, "images", target_subset)
    mask_dir = os.path.join(data_root, split, "masks_pixel_gt", target_subset)
    
    if not os.path.exists(img_dir):
        print(f"[ERROR] ❌ {split} set path NOT found: {img_dir}")
        print("建议: 请检查 data/PRMI 下是否真的有 train/val/test 文件夹")
    else:
        files = glob.glob(os.path.join(img_dir, "*"))
        print(f"[OK] ✅ {split} set found: {len(files)} images.")