# Project 4 — Path 1: Optical Character Recognition (OCR)
### DecodeLabs AI Industrial Training Kit | Batch 2026

---

## Overview

This project implements a **text recognition pipeline** using Google's Tesseract OCR engine wrapped with pytesseract and pre-processed through OpenCV. It extracts machine-readable text from raw images with confidence-filtered output.

---

## Objective

Engineer a Python script capable of ingesting raw visual data and extracting accurate, machine-readable text strings — proving the machine can **read** with validated confidence.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| pytesseract | Python wrapper for Tesseract OCR engine |
| Tesseract v5.x | Google's OCR engine (CNN + LSTM pipeline) |
| OpenCV (`cv2`) | Image pre-processing |
| NumPy | Array and matrix operations |
| Pillow | Image format support |

---

## How It Works

### The Pipeline

```
Input Image
    ↓
Step 1: Load Image (cv2.imread)
        - Validate file exists
        - Read as 3D array (H x W x C)
    ↓
Step 2: Grayscale Conversion (cv2.cvtColor)
        - Collapse 3D RGB matrix → 1D intensity matrix
        - Remove distracting colour data
    ↓
Step 3: Gaussian Blur (cv2.GaussianBlur)
        - Smooth image to eliminate artifact noise
        - Kernel size: 3x3
    ↓
Step 4: Otsu's Thresholding (cv2.threshold)
        - Force every pixel to choose: 0 (black) or 255 (white)
        - Auto-calculates optimal cutoff value
    ↓
Step 5: Deskewing
        - Calculate rotation angle using pixel moments
        - Snap tilted text back to horizontal baseline
    ↓
Step 6: OCR via pytesseract
        - Convolutional + Bi-directional LSTM pipeline
        - Returns word-level text with confidence scores
    ↓
Step 7: Confidence Filter (>= 80% gate)
        - Words below 80% confidence are dropped
    ↓
Output: Extracted text + confidence report + stage images saved
```

### Key Concepts

**Grayscale Conversion** — A colour image is a 3D matrix (Height × Width × 3 channels). Converting to grayscale collapses it to a 2D intensity matrix, removing colour noise that confuses the OCR engine.

**Otsu's Thresholding** — Forces a binary decision on every pixel. Instead of shades of grey, each pixel becomes pure black or white. Otsu's method automatically finds the optimal cutoff value by analysing the histogram of pixel intensities.

**Deskewing** — Real-world images are often slightly tilted. This step calculates the rotation angle using pixel coordinate moments and rotates the image back to a perfect horizontal baseline, dramatically improving OCR accuracy.

**PSM (Page Segmentation Mode)** — Tesseract needs to know the layout of text in the image. Different PSM modes tell it whether to expect a full page, a single line, scattered text, etc.

**80% Confidence Gate** — Tesseract assigns a confidence score (0–100) to every word it reads. Only words scoring ≥ 80% are included in the final output, filtering out uncertain guesses.

---

## Setup

### Step 1 — Install Python dependencies
```bash
pip install pytesseract opencv-python Pillow numpy
```

### Step 2 — Install Tesseract OCR engine

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the `.exe` and install with default settings
3. Add this line to `ocr_recognition.py` after the imports:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Step 3 — Verify Tesseract is installed
```bash
tesseract --version
```
You should see `tesseract v5.x.x`.

---

## Usage

```bash
python ocr_recognition.py <image_path>
```

### Examples
```bash
# Basic usage
python ocr_recognition.py test.png

# Single text block (book pages, paragraphs)
python ocr_recognition.py test.png --psm 6

# Single line (number plates, headers)
python ocr_recognition.py test.png --psm 7

# Scattered text (invoices, receipts)
python ocr_recognition.py test.png --psm 11

# Adjust confidence threshold
python ocr_recognition.py test.png --confidence 85

# Custom output folder
python ocr_recognition.py test.png --output my_results
```

## Sample Output

```
>>> DecodeLabs Project 4 — Path 1: OCR Pipeline

[✓] Image loaded  →  shape: (1080, 1920, 3)  (H x W x C)
[✓] Grayscale applied  →  shape: (1080, 1920)
[✓] Gaussian Blur applied  →  kernel: 3x3
[✓] Otsu Thresholding applied  →  binary image ready
[✓] Deskew: angle -0.04° — image is already straight
[✓] Stage images saved → ocr_output/

============================================================
  OCR RECOGNITION RESULTS
============================================================

  Confidence Threshold : 80.0%
  Words Accepted       : 96
  Average Confidence   : 93.36%
  Accuracy Gate        : ✅ PASS

  ── Extracted Text ───────────────────────────────

  YouTube  Ask Gemini
  AI  Podcasts  Music  JavaScript  Gaming
  Build Your First AI Agent
  Deploy
  Simplify DB operations with Cloud SQL gemini
  ...

  ── High-Confidence Words ────────────────────────
  Gaming               96.0%  █████████
  Git                  96.0%  █████████
  JavaScript           80.0%  ████████
  YouTube              95.0%  █████████
  ...
============================================================
```

---

## Output Files

After running, the `ocr_output/` folder contains:

| File | Description |
|------|-------------|
| `1_original.jpg` | Original input image |
| `2_grayscale.jpg` | After grayscale conversion |
| `3_threshold.jpg` | After Otsu's thresholding (binary) |

These demonstrate the pre-processing pipeline visually.

---

## Validation Checkpoints

| # | Checkpoint | Status |
|---|-----------|--------|
| 1 | Library Integration | ✅ pytesseract running error-free |
| 2 | Pre-Processing Integrity | ✅ Grayscale → Blur → Threshold → Deskew executed |
| 3 | Accuracy Benchmarking | ✅ Average confidence 93.36% (≥ 80%) |
| 4 | Visual Confirmation | ✅ 96 words extracted, stage images saved |

## Author
**Harsh Vardhan Singh** | DecodeLabs AI Internship | Batch 2026