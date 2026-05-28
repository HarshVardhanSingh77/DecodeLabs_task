"""
Project 4 - Path 2: Object Detection with MobileNet-SSD
DecodeLabs AI Industrial Training Kit | Batch 2026

Pipeline:
  1. Load image
  2. Blob construction (blobFromImage — 4D tensor for the network)
  3. Forward pass through MobileNet-SSD (cv2.dnn)
  4. Decode detections: confidence filter (>= 80%), bounding box scaling
  5. Draw bounding boxes + labels on output image
  6. Save visual confirmation
"""

import cv2
import numpy as np
import os
import sys
import argparse

# ─────────────────────────────────────────────
#  CLASS LABELS (MobileNet-SSD trained on VOC — 20 classes)
# ─────────────────────────────────────────────
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")

# ─────────────────────────────────────────────
#  MODEL PATHS
# ─────────────────────────────────────────────
MODEL_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
PROTOTXT_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_PATH    = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")


def check_model_files():
    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
        print("\n[✗] Model files not found!")
        print("    Run this first:  python download_model.py\n")
        sys.exit(1)


# ─────────────────────────────────────────────
#  STEP 1: LOAD IMAGE
# ─────────────────────────────────────────────
def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"OpenCV could not read: {path}")
    print(f"[✓] Image loaded  →  {img.shape[1]}x{img.shape[0]} px")
    return img


# ─────────────────────────────────────────────
#  STEP 2: LOAD MODEL
# ─────────────────────────────────────────────
def load_model() -> cv2.dnn_Net:
    print("[→] Loading MobileNet-SSD model ...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    print("[✓] Model loaded (Transfer Learning: ImageNet → VOC)")
    return net


# ─────────────────────────────────────────────
#  STEP 3: BLOB CONSTRUCTION (4D Tensor)
# ─────────────────────────────────────────────
def build_blob(img: np.ndarray) -> np.ndarray:
    """
    Convert image to the 4D blob the network expects.
    blobFromImage performs:
      - Scaling to 300x300 (required network input size)
      - Mean subtraction (127.5) to normalize pixel values
    """
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=127.5
    )
    print(f"[✓] Blob constructed  →  shape: {blob.shape}  (N x C x H x W)")
    return blob


# ─────────────────────────────────────────────
#  STEP 4: FORWARD PASS + DETECTION DECODE
# ─────────────────────────────────────────────
def detect_objects(net, blob, img, confidence_threshold=0.80):
    net.setInput(blob)
    detections = net.forward()

    (h, w) = img.shape[:2]
    results = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue

        class_id = int(detections[0, 0, i, 1])
        label    = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"

        # Scale normalised coords → pixel space
        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w, int(detections[0, 0, i, 5] * w))
        y2 = min(h, int(detections[0, 0, i, 6] * h))

        results.append({
            "label":      label,
            "class_id":   class_id,
            "confidence": round(confidence * 100, 2),
            "box":        (x1, y1, x2 - x1, y2 - y1),
        })

    print(f"[✓] Detections >= {confidence_threshold*100:.0f}% confidence: {len(results)}")
    return results


# ─────────────────────────────────────────────
#  STEP 5: DRAW BOUNDING BOXES
# ─────────────────────────────────────────────
def draw_detections(img, detections):
    output = img.copy()
    for det in detections:
        (x, y, bw, bh) = det["box"]
        color      = [int(c) for c in COLORS[det["class_id"] % len(COLORS)]]
        label_text = f"{det['label']}  {det['confidence']}%"

        cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 2)

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = y - 10 if y - 10 > th else y + th + 10
        cv2.rectangle(output, (x, label_y - th - 4), (x + tw + 4, label_y + 4), color, -1)
        cv2.putText(output, label_text, (x + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return output


# ─────────────────────────────────────────────
#  DISPLAY RESULTS
# ─────────────────────────────────────────────
def display_results(detections, confidence_threshold):
    print("\n" + "=" * 60)
    print("  OBJECT DETECTION RESULTS")
    print("=" * 60)
    print(f"\n  Confidence Threshold : {confidence_threshold * 100:.0f}%")
    print(f"  Objects Detected     : {len(detections)}")

    if not detections:
        print("\n  No objects detected above the confidence threshold.")
        print("\n  Tip: Try lowering threshold with --confidence 0.5")
    else:
        print(f"\n  {'Object':<20} {'Conf':>7}  Bounding Box (X,Y,W,H)")
        print(f"  {'-'*20}  {'-'*7}  {'-'*28}")
        for d in detections:
            print(f"  {'OK'} {d['label']:<18} {d['confidence']:>6.1f}%  {d['box']}")

    gate = "PASS" if len(detections) > 0 else "NO DETECTIONS (try lower threshold)"
    print(f"\n  Accuracy Gate        : {gate}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(image_path, confidence_threshold=0.80, output_path="detection_output.jpg"):
    print("\n>>> DecodeLabs Project 4 - Path 2: Object Detection Pipeline\n")

    check_model_files()

    img        = load_image(image_path)
    net        = load_model()
    blob       = build_blob(img)
    detections = detect_objects(net, blob, img, confidence_threshold)
    output_img = draw_detections(img, detections)

    cv2.imwrite(output_path, output_img)
    print(f"[✓] Output saved → {output_path}")

    display_results(detections, confidence_threshold)
    return detections


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project 4 Path 2 - Object Detection")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--confidence", type=float, default=0.80,
                        help="Min confidence 0.0 to 1.0 (default: 0.80)")
    parser.add_argument("--output", default="detection_output.jpg",
                        help="Output image path")
    args = parser.parse_args()

    run_pipeline(args.image, args.confidence, args.output)