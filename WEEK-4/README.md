# Project 4: Image & Text Recognition
### DecodeLabs AI Industrial Training Kit | Batch 2026

---

## What This Project Does

You have **two execution paths** — pick one or run both:

| | Path 1: OCR | Path 2: Object Detection |
|---|---|---|
| Goal | Extract text from images | Find & locate objects |
| Library | `pytesseract` | `cv2.dnn` + MobileNet-SSD |
| Pre-processing | Grayscale → Blur → Threshold → Deskew | Blob construction (300×300) |
| Output | Formatted text strings | Bounding boxes (X, Y, W, H) |
| Confidence Gate | ≥ 80% per word | ≥ 80% per detection |

---

## Setup

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Install Tesseract OCR engine (Path 1 only)

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr
```

**Windows:**
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
Then add to PATH, or set in your script:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**macOS:**
```bash
brew install tesseract
```

### Step 3 — Download MobileNet-SSD model (Path 2 only)
```bash
cd path2_object_detection
python download_model.py
```
This auto-downloads the `.pbtxt` config and `frozen_inference_graph.pb` weights (~28 MB).

---

## Running Path 1: OCR

```bash
cd path1_ocr
python ocr_recognition.py path/to/your/image.jpg
```

**Options:**
```bash
# Change PSM mode (layout type)
python ocr_recognition.py image.jpg --psm 6        # single text block
python ocr_recognition.py image.jpg --psm 7        # single line (number plate)
python ocr_recognition.py image.jpg --psm 11       # scattered text (invoice)

# Adjust confidence threshold
python ocr_recognition.py image.jpg --confidence 85

# Choose output folder for stage images
python ocr_recognition.py image.jpg --output my_results
```

**PSM Quick Reference:**
| PSM | Best For |
|-----|----------|
| 3 | General documents (default) |
| 6 | Books, paragraphs |
| 7 | Single lines, number plates |
| 11 | Invoices, receipts, scattered text |

---

## Running Path 2: Object Detection

```bash
cd path2_object_detection

# First, download the model (only once)
python download_model.py

# Then run detection
python object_detection.py path/to/your/image.jpg
```

**Options:**
```bash
# Save output to custom path
python object_detection.py image.jpg --output result.jpg

# Lower confidence threshold (catch more objects)
python object_detection.py image.jpg --confidence 0.60

# Specify model files manually
python object_detection.py image.jpg \
  --config model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt \
  --weights model/frozen_inference_graph.pb
```

---

## Expected Output

### Path 1 (OCR)
```
>>> DecodeLabs Project 4 — Path 1: OCR Pipeline

[✓] Image loaded  →  shape: (480, 640, 3)  (H x W x C)
[✓] Grayscale applied  →  shape: (480, 640)
[✓] Gaussian Blur applied  →  kernel: 3x3
[✓] Otsu Thresholding applied  →  binary image ready
[✓] Deskew: corrected by -2.45°
[✓] Stage images saved → ocr_output/

============================================================
  OCR RECOGNITION RESULTS
============================================================

  Confidence Threshold : 80.0%
  Words Accepted       : 12
  Average Confidence   : 91.6%
  Accuracy Gate        : ✅ PASS

  ── Extracted Text ───────────────────────────────

  Invoice #0042
  Date: 2023-10-27
  Total: $499.00

  ── High-Confidence Words ────────────────────────
  Invoice              96.4%  █████████
  0042                 88.1%  ████████
  ...
```

### Path 2 (Object Detection)
```
>>> DecodeLabs Project 4 — Path 2: Object Detection Pipeline

[✓] Image loaded  →  640x480 px
[✓] Model loaded (Transfer Learning: ImageNet → COCO)
[✓] Blob constructed  →  shape: (1, 3, 300, 300)
[✓] Detections ≥ 80% confidence: 3
[✓] Output saved → detection_output.jpg

============================================================
  OBJECT DETECTION RESULTS
============================================================

  Confidence Threshold : 80%
  Objects Detected     : 3

  ── Detections ───────────────────────────────────
  Object               Conf    Bounding Box (X,Y,W,H)
  --------------------  -------  ----------------------------
  ✅ person             91.2%  (45, 30, 180, 310)
  ✅ car                88.5%  (300, 120, 240, 160)
  ✅ dog                83.1%  (10, 200, 90, 100)

  Accuracy Gate        : ✅ PASS
```

---

## The 4 Validation Checkpoints

| # | Checkpoint | How It's Met |
|---|-----------|-------------|
| 1 | Library Integration | `pytesseract` / `cv2.dnn` used without errors |
| 2 | Pre-Processing Integrity | Grayscale → Blur → Threshold pipeline runs & saves stages |
| 3 | Accuracy Benchmarking | 80% confidence gate applied and printed |
| 4 | Visual Confirmation | Stage images saved (OCR) / annotated image saved (Detection) |

---

## Project Structure
```
project4/
├── requirements.txt
├── README.md
├── path1_ocr/
│   └── ocr_recognition.py
└── path2_object_detection/
    ├── object_detection.py
    ├── download_model.py
    └── model/               ← created by download_model.py
        ├── *.pbtxt
        └── frozen_inference_graph.pb
```
