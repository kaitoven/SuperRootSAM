import os
import requests
import sys

def download_weight():
    save_dir = "checkpoints"
    filename = "sam2.1_hiera_large.pt"
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    save_path = os.path.join(save_dir, filename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(save_path):
        print(f"[Info] File exists: {save_path}")
        return

    print(f"[Start] Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        wrote = 0
        with open(save_path, 'wb') as f:
            for data in response.iter_content(1024*1024):
                wrote += len(data)
                f.write(data)
                if total_size > 0:
                    sys.stdout.write(f"\rProgress: {wrote/total_size*100:.2f}%")
                    sys.stdout.flush()
        print(f"\n[Success] Saved to {save_path}")
    except Exception as e:
        print(f"\n[Error] {e}")

if __name__ == "__main__":
    download_weight()