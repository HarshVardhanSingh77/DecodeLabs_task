# DecodeLabs AI Internship
### Artificial Intelligence Engineer Track | Batch 2026

---

## Intern Details

| Field | Details |
|-------|---------|
| **Name** | Harsh Vardhan Singh |
| **Track** | Artificial Intelligence |
| **Batch** | 2026 |
| **Organisation** | DecodeLabs |

---

## Internship Overview

This internship follows a structured learning path from foundational AI concepts to real-world model implementation. Each week builds on the previous, progressing from rule-based logic to classical ML, recommendation systems, and finally computer vision.

```
WEEK 1          WEEK 2          WEEK 3          WEEK 4
  ↓               ↓               ↓               ↓
Rule-Based      KNN &         Recommender     Image & Text
AI Chatbot    Dashboard        Systems        Recognition
(Pure Python)  (scikit-learn)  (Filtering)   (CV + OCR)
```

---

## Repository Structure

```
Decodelabs Internship/
├── README.md                            ← This file
│
├── WEEK-1/                              ← Rule-Based AI
│   ├── Chatbot.py
│   └── README.md
│
├── WEEK-2/                              ← Classical ML
│   ├── iris_knn_project.py
│   ├── iris_knn_dashboard.py
│   └── README.md
│
├── WEEK-3/                              ← Recommendation Systems
│   ├── recommender.py
│   ├── raw_skills.csv
│   └── README.md
│
└── WEEK-4/                              ← Computer Vision (Project 4)
    ├── path1_ocr/
    │   ├── ocr_recognition.py
    │   ├── test.png
    │   ├── README.md
    │   └── ocr_output/
    └── path2_object_detection/
        ├── object_detection.py
        ├── download_model.py
        ├── test.png
        ├── detection_output.jpg
        ├── README.md
        └── model/
```

---

## Week-by-Week Progress

---

### WEEK 1 — Rule-Based AI Chatbot

**Goal:** Build a conversational chatbot using pure Python if-else logic with no external libraries.

**Project:** `Chatbot.py` — Advanced Rule-Based AI Chatbot named **DecodeBot**

**How It Works:**
```
User types input
      ↓
Input converted to lowercase
      ↓
Matched against if-elif conditions
      ↓
Matching response printed by bot
      ↓
Loop continues until 'bye' or 'exit'
```

**Supported Categories:**
- Greetings (hello, good morning, good evening)
- About the bot (name, creator, type)
- Health & feelings (happy, sad, tired)
- Education (what is Python, AI, ML, chatbot)
- Fun responses (jokes, facts, motivation)
- System commands (help, bye, exit)

**Key Concepts:**
- Rule-Based AI and heuristic logic
- `while True` loop for continuous conversation
- `.lower()` for case-insensitive string matching
- `break` statement for clean exit

**Tech Stack:** Pure Python 3 — no external libraries

**Sample Run:**
```
============================================
🤖 ADVANCED RULE-BASED AI CHATBOT
============================================
You: hi
Bot: Hello! Nice to meet you 😊
You: motivate me
Bot: Success starts with consistency and hard work 💪
You: bye
Bot: Goodbye! Have a great day 👋
```

**Status:** ✅ Complete

---

### WEEK 2 — Classical Machine Learning (KNN)

**Goal:** Build and visualise a K-Nearest Neighbours classifier on the Iris dataset.

**Projects:**
- `iris_knn_project.py` — KNN classifier implementation
- `iris_knn_dashboard.py` — Interactive visual dashboard

**Key Concepts:**
- K-Nearest Neighbours algorithm
- Training and testing splits
- Model accuracy evaluation
- Data visualisation

**Tech Stack:** Python, scikit-learn, pandas, matplotlib

**Status:** ✅ Complete

---

### WEEK 3 — Recommendation Systems

**Goal:** Build a content-based recommendation engine using skill data.

**Projects:**
- `recommender.py` — Recommendation engine
- `raw_skills.csv` — Skills dataset

**Key Concepts:**
- Content-based filtering
- Cosine similarity
- Feature extraction from raw data
- Building recommendation pipelines

**Tech Stack:** Python, pandas, scikit-learn, NumPy

**Status:** ✅ Complete

---

### WEEK 4 — Computer Vision: Image & Text Recognition (Project 4)

**Goal:** Implement a basic image or text recognition task using pre-trained AI libraries — demonstrating the ability to integrate pre-trained models into a functional workflow.

This is the **Optional Mastery Phase** of the internship, demonstrating model implementation skills beyond the core curriculum.

---

#### Path 1: Optical Character Recognition (OCR)

**Goal:** Extract machine-readable text from raw images using pytesseract.

**Pipeline:**
```
Image → Grayscale → Gaussian Blur → Otsu Thresholding → Deskew → OCR → Confidence Filter
```

**Results:**
- Words Extracted: 96
- Average Confidence: 93.36%
- Accuracy Gate: ✅ PASS (≥ 80%)

**Tech Stack:** Python, OpenCV, pytesseract, Tesseract v5, NumPy, Pillow

---

#### Path 2: Object Detection with MobileNet-SSD

**Goal:** Identify and locate physical objects in images using a pre-trained deep learning model.

**Pipeline:**
```
Image → Blob Construction (300x300) → MobileNet-SSD Forward Pass → Confidence Filter → Bounding Boxes
```

**Results:**
- Objects Detected: 3 (bus 98.6%, person 92.3%, person 86.6%)
- Accuracy Gate: ✅ PASS (≥ 80%)

**Tech Stack:** Python, OpenCV (cv2.dnn), MobileNet-SSD (Caffe), NumPy

---

#### Week 4 Validation Checkpoints

| # | Checkpoint | Path 1 OCR | Path 2 Detection |
|---|-----------|-----------|-----------------|
| 1 | Library Integration | ✅ | ✅ |
| 2 | Pre-Processing Integrity | ✅ | ✅ |
| 3 | Accuracy Benchmarking ≥ 80% | ✅ 93.36% | ✅ 92.3% avg |
| 4 | Visual Confirmation | ✅ | ✅ |

**Status:** ✅ Complete

---

## Skills Demonstrated

| Skill | Where Applied |
|-------|--------------|
| Python programming | All weeks |
| Rule-based logic & if-else | Week 1 |
| String manipulation | Week 1 |
| Loop control & flow | Week 1 |
| Data preprocessing | Week 2, 3, 4 |
| Classical ML (KNN) | Week 2 |
| Model evaluation & accuracy | Week 2, 4 |
| Data visualisation | Week 2 |
| Recommendation systems | Week 3 |
| Cosine similarity | Week 3 |
| OpenCV image processing | Week 4 |
| OCR & text extraction | Week 4 Path 1 |
| Deep learning inference (DNN) | Week 4 Path 2 |
| Transfer learning | Week 4 Path 2 |
| Confidence thresholding | Week 4 |
| Bounding box decoding | Week 4 Path 2 |

---

## Overall Progress

| Week | Project | Status |
|------|---------|--------|
| Week 1 | Rule-Based AI Chatbot (DecodeBot) | ✅ Complete |
| Week 2 | KNN Classifier & Dashboard | ✅ Complete |
| Week 3 | Recommendation System | ✅ Complete |
| Week 4 | OCR + Object Detection (Project 4) | ✅ Complete |

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Week 1 — No dependencies needed
```bash
cd "WEEK-1"
python Chatbot.py
```

### Week 2 & 3
```bash
pip install numpy pandas scikit-learn matplotlib
```

### Week 4 — All dependencies
```bash
pip install opencv-python pytesseract Pillow numpy

# Install Tesseract engine (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki

# Download MobileNet-SSD model
cd WEEK-4/path2_object_detection
python download_model.py
```

---

## How to Run Each Project

### Week 1 — Chatbot
```bash
cd WEEK-1
python Chatbot.py
```

### Week 2 — KNN
```bash
cd WEEK-2
python iris_knn_project.py
python iris_knn_dashboard.py
```

### Week 3 — Recommender
```bash
cd WEEK-3
python recommender.py
```

### Week 4 — OCR
```bash
cd WEEK-4/path1_ocr
python ocr_recognition.py test.png
```

### Week 4 — Object Detection
```bash
cd WEEK-4/path2_object_detection
python download_model.py        # run once
python object_detection.py test.png
```

---

## Author

**Harsh Vardhan Singh**
DecodeLabs AI Internship | Batch 2026
