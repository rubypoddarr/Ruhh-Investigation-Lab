"""
Download YOLOv3 weights file.
Run this script once before starting the app:
    python download_weights.py
The file is ~248 MB and is required for object detection.
"""
import os
import urllib.request

WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolov3.weights")

def download_with_progress(url, dest):
    """Download a file and show progress."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    print(f"Downloading YOLOv3 weights (~248 MB)...")
    print(f"URL: {url}")
    print(f"Destination: {dest}")
    print("")

    def reporthook(count, block_size, total_size):
        downloaded = count * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:3d}%  ({mb_done:.1f} / {mb_total:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print("\n\nDownload complete!")
    print(f"Saved to: {dest}")

if __name__ == "__main__":
    if os.path.exists(WEIGHTS_PATH):
        size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
        print(f"yolov3.weights already exists ({size_mb:.1f} MB). Skipping download.")
    else:
        download_with_progress(WEIGHTS_URL, WEIGHTS_PATH)
        print("\nYou can now run: python app.py")
