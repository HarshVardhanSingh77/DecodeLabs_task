# Project 4 — Path 2: Object Detection with MobileNet-SS

## Overview

This project implements a **real-time object detection pipeline** using a pre-trained MobileNet-SSD model loaded through OpenCV's DNN module. It detects and locates physical objects in images, drawing labelled bounding boxes with confidence scores.

---

## Objective

Engineer a Python script capable of ingesting raw visual data and identifying physical entities with their spatial coordinates — proving the machine can **see** objects with validated confidence.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| OpenCV (`cv2.dnn`) | DNN inference engine |
| MobileNet-SSD (Caffe) | Pre-trained object detection model |
| NumPy | Array and matrix operations |

---

## How It Works

### The Pipeline

```
Input Image
    ↓
Step 1: Load Image (cv2.imread)
    ↓
Step 2: Load MobileNet-SSD Model (cv2.dnn.readNetFromCaffe)
    ↓
Step 3: Blob Construction (cv2.dnn.blobFromImage)
        - Resize to 300x300
        - Mean subtraction (127.5)
        - Normalize pixel values
    ↓
Step 4: Forward Pass (net.forward())
        - Network outputs normalized coordinates
        - Each detection: [class_id, confidence, x1, y1, x2, y2]
    ↓
Step 5: Confidence Filter (>= 80% gate)
        - Detections below 80% are dropped
    ↓
Step 6: Bounding Box Scaling
        - Multiply normalized coords by image width/height
    ↓
Step 7: Draw boxes + labels on image
    ↓
Output: Annotated image saved as detection_output.jpg
```

### Key Concepts

**Transfer Learning** — MobileNet-SSD was pre-trained on ImageNet (millions of images). We reuse this knowledge directly without retraining, achieving high accuracy with zero training compute.

**Blob Construction** — The network doesn't take a raw image. `cv2.dnn.blobFromImage` converts it into a 4D tensor `(N x C x H x W)` = `(1, 3, 300, 300)`, performs mean subtraction to remove lighting bias, and normalizes pixel values.

**Bounding Box Coordinates** — The model outputs normalized coordinates between 0 and 1. We multiply by the actual image pixel dimensions to get real pixel-space boxes `(X, Y, W, H)`.

**80% Confidence Gate** — Every detection comes with a Softmax confidence score. Only detections ≥ 80% are accepted, preventing false positives.

---

## Setup

### Step 1 — Install dependencies
```bash
pip install opencv-python numpy
```

### Step 2 — Download model files (run once)
```bash
python download_model.py
```
This downloads:
- `MobileNetSSD_deploy.prototxt` — network architecture config
- `MobileNetSSD_deploy.caffemodel` — pre-trained weights (~22 MB)

---

## Usage

```bash
python object_detection.py <image_path>
```

### Examples
```bash
# Basic usage
python object_detection.py test.png

# Lower confidence threshold to detect more objects
python object_detection.py test.png --confidence 0.60

# Save output to custom path
python object_detection.py test.png --output result.jpg
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `image` | required | Path to input image |
| `--confidence` | `0.80` | Minimum confidence threshold (0.0 to 1.0) |
| `--output` | `detection_output.jpg` | Path to save annotated output image |

---

## Sample Output

```
>>> DecodeLabs Project 4 - Path 2: Object Detection Pipeline

[✓] Image loaded  →  1920x1080 px
[✓] Model loaded (Transfer Learning: ImageNet → VOC)
[✓] Blob constructed  →  shape: (1, 3, 300, 300)  (N x C x H x W)
[✓] Detections >= 80% confidence: 3
[✓] Output saved → detection_output.jpg

============================================================
  OBJECT DETECTION RESULTS
============================================================

  Confidence Threshold : 80%
  Objects Detected     : 3

  Object                  Conf    Bounding Box (X,Y,W,H)
  --------------------  -------  ----------------------------
  OK bus                  98.6%  (0, 0, 1910, 1080)
  OK person               92.3%  (1467, 282, 250, 280)
  OK person               86.6%  (1107, 280, 181, 344)

  Accuracy Gate        : PASS
============================================================
```

---

## Detectable Object Classes (20 classes)

`aeroplane` `bicycle` `bird` `boat` `bottle` `bus` `car` `cat` `chair` `cow` `diningtable` `dog` `horse` `motorbike` `person` `pottedplant` `sheep` `sofa` `train` `tvmonitor`

---

## Validation Checkpoints

| # | Checkpoint | Status |
|---|-----------|--------|
| 1 | Library Integration | ✅ cv2.dnn running error-free |
| 2 | Pre-Processing Integrity | ✅ Blob (1x3x300x300) constructed |
| 3 | Accuracy Benchmarking | ✅ All detections ≥ 80% confidence |
| 4 | Visual Confirmation | ✅ Annotated image saved with bounding boxes |

## Author
**Harsh Vardhan Singh** | DecodeLabs AI Internship | Batch 2026