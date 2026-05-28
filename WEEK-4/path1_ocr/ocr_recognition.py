"""
Project 4 - Path 1: Optical Character Recognition (OCR)
DecodeLabs AI Industrial Training Kit | Batch 2026

Pipeline:
  1. Load image
  2. Grayscale conversion (removes chromatic noise)
  3. Gaussian Blur (smooths micro-imperfections)
  4. Adaptive Thresholding / Otsu's Method (forces binary decision)
  5. Deskewing (corrects tilted text)
  6. OCR via pytesseract
  7. Confidence-filtered output (>= 80% gate)
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import sys
import argparse
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ─────────────────────────────────────────────
#  STEP 1: IMAGE LOADING
# ─────────────────────────────────────────────
def load_image(path: str) -> np.ndarray:
    """Load image from disk and validate it."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"OpenCV could not read the image: {path}")
    print(f"[✓] Image loaded  →  shape: {img.shape}  (H x W x C)")
    return img


# ─────────────────────────────────────────────
#  STEP 2: GRAYSCALE CONVERSION
# ─────────────────────────────────────────────
def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Collapse 3D RGB matrix → 1D intensity matrix."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"[✓] Grayscale applied  →  shape: {gray.shape}")
    return gray


# ─────────────────────────────────────────────
#  STEP 3: GAUSSIAN BLUR
# ─────────────────────────────────────────────
def apply_blur(gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Smooth image to eliminate artifact noise."""
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    print(f"[✓] Gaussian Blur applied  →  kernel: {kernel_size}x{kernel_size}")
    return blurred


# ─────────────────────────────────────────────
#  STEP 4: ADAPTIVE THRESHOLDING (Otsu's Method)
# ─────────────────────────────────────────────
def apply_threshold(blurred: np.ndarray) -> np.ndarray:
    """
    Force every pixel to choose a side (0=black or 255=white).
    Uses Otsu's method to automatically find the optimal cutoff.
    """
    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"[✓] Otsu Thresholding applied  →  binary image ready")
    return thresh


# ─────────────────────────────────────────────
#  STEP 5: DESKEWING
# ─────────────────────────────────────────────
def deskew(thresh: np.ndarray) -> np.ndarray:
    """
    Calculate rotation angle and snap tilted text back to horizontal.
    Uses pixel coordinate moments to find the skew angle.
    """
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 5:
        print("[!] Not enough pixels to deskew — skipping")
        return thresh

    angle = cv2.minAreaRect(coords)[-1]

    # Correct angle convention
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        print(f"[✓] Deskew: angle {angle:.2f}° — image is already straight")
        return thresh

    (h, w) = thresh.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(thresh, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    print(f"[✓] Deskew: corrected by {angle:.2f}°")
    return deskewed


# ─────────────────────────────────────────────
#  STEP 6: OCR + CONFIDENCE FILTER (≥ 80%)
# ─────────────────────────────────────────────
def run_ocr(processed_img: np.ndarray, psm: int = 3,
             confidence_threshold: float = 80.0) -> dict:
    """
    Run pytesseract OCR and filter results by confidence score.

    PSM Modes:
      3  - Fully automatic (default, varied layouts)
      6  - Single uniform block of text (book pages)
      7  - Single text line (number plates / headers)
      11 - Sparse, scattered text (invoices)
    """
    config = f"--psm {psm} --oem 3"

    # Get detailed word-level data
    data = pytesseract.image_to_data(
        processed_img,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    words, confidences = [], []
    for i, conf in enumerate(data["conf"]):
        try:
            conf_val = float(conf)
        except (ValueError, TypeError):
            continue
        if conf_val >= confidence_threshold and data["text"][i].strip():
            words.append(data["text"][i])
            confidences.append(conf_val)

    full_text = pytesseract.image_to_string(processed_img, config=config).strip()

    # Calculate average confidence of accepted words
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    result = {
        "raw_text":          full_text,
        "filtered_words":    words,
        "confidences":       confidences,
        "average_confidence": round(avg_conf, 2),
        "word_count":        len(words),
        "threshold_used":    confidence_threshold,
    }
    return result


# ─────────────────────────────────────────────
#  STEP 7: SAVE INTERMEDIATE RESULTS
# ─────────────────────────────────────────────
def save_stages(original, gray, thresh, output_dir: str):
    """Save preprocessing stage images for visual confirmation."""
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "1_original.jpg"), original)
    cv2.imwrite(os.path.join(output_dir, "2_grayscale.jpg"), gray)
    cv2.imwrite(os.path.join(output_dir, "3_threshold.jpg"), thresh)
    print(f"[✓] Stage images saved → {output_dir}/")


# ─────────────────────────────────────────────
#  DISPLAY OUTPUT
# ─────────────────────────────────────────────
def display_results(result: dict):
    print("\n" + "=" * 60)
    print("  OCR RECOGNITION RESULTS")
    print("=" * 60)
    print(f"\n  Confidence Threshold : {result['threshold_used']}%")
    print(f"  Words Accepted       : {result['word_count']}")
    print(f"  Average Confidence   : {result['average_confidence']}%")
    status = "✅ PASS" if result['average_confidence'] >= 80 else "❌ FAIL (< 80%)"
    print(f"  Accuracy Gate        : {status}")
    print("\n  ── Extracted Text ───────────────────────────────")
    print(f"\n{result['raw_text']}\n")

    if result['filtered_words']:
        print("  ── High-Confidence Words ────────────────────────")
        for word, conf in zip(result['filtered_words'], result['confidences']):
            bar = "█" * int(conf // 10)
            print(f"  {word:<20} {conf:>5.1f}%  {bar}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(image_path: str, psm: int = 3,
                 output_dir: str = "ocr_output",
                 confidence_threshold: float = 80.0):
    print("\n>>> DecodeLabs Project 4 — Path 1: OCR Pipeline\n")

    # Stage 1: Load
    original = load_image(image_path)

    # Stage 2: Grayscale
    gray = to_grayscale(original)

    # Stage 3: Blur
    blurred = apply_blur(gray)

    # Stage 4: Threshold
    thresh = apply_threshold(blurred)

    # Stage 5: Deskew
    deskewed = deskew(thresh)

    # Stage 6: OCR + confidence filter
    result = run_ocr(deskewed, psm=psm,
                     confidence_threshold=confidence_threshold)

    # Stage 7: Save + display
    save_stages(original, gray, thresh, output_dir)
    display_results(result)

    return result


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project 4 Path 1 — OCR Recognition Pipeline"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--psm", type=int, default=3,
                        help="Tesseract Page Segmentation Mode (default: 3)")
    parser.add_argument("--confidence", type=float, default=80.0,
                        help="Minimum confidence threshold  (default: 80)")
    parser.add_argument("--output", default="ocr_output",
                        help="Directory to save stage images")
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        psm=args.psm,
        output_dir=args.output,
        confidence_threshold=args.confidence
    )
