"""
download_model.py — Download MobileNet-SSD model files
Run this ONCE before using object_detection.py
"""
import os
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = [
    {
        "name": "MobileNet-SSD prototxt (config)",
        "url": "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt",
        "path": os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt"),
    },
    {
        "name": "MobileNet-SSD caffemodel (weights ~22MB)",
        "url": "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.caffemodel",
        "path": os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel"),
    },
]

print("\n>>> Downloading MobileNet-SSD model files...\n")

for f in FILES:
    if os.path.exists(f["path"]):
        print(f"[✓] Already exists → {f['path']}")
        continue
    print(f"[→] Downloading {f['name']} ...")
    try:
        urllib.request.urlretrieve(f["url"], f["path"])
        size = os.path.getsize(f["path"])
        print(f"[✓] Saved ({size // 1024} KB) → {f['path']}")
    except Exception as e:
        print(f"[✗] Failed: {e}")

print("\n[✓] Model files ready. You can now run object_detection.py\n")
