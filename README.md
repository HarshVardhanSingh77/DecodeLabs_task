# 🌸 Project 2: Data Classification Using AI


## 📌 Overview

This project implements a **K-Nearest Neighbors (KNN)** classification model on the classic **Iris dataset**. It covers the complete supervised learning pipeline — from raw data to intelligent predictions — including feature scaling, train/test splitting, model tuning, evaluation, and visualization.

---

## 🎯 Goal

Build a basic classification model that can identify the species of an Iris flower based on its physical measurements.

---

## 📁 Project Structure

```
decodelabs_project2/
│
├── iris_knn_project.py       # Main project code
├── iris_knn_dashboard.png    # Output visualization dashboard
├── README.md                 # Project documentation
└── venv/                     # Virtual environment (auto-generated)
```

---

## 🗃️ Dataset: The Iris Benchmark

| Property   | Details                                      |
|------------|----------------------------------------------|
| Samples    | 150 (balanced — 50 per class)                |
| Classes    | 3 (Setosa, Versicolor, Virginica)            |
| Features   | 4 (Sepal Length, Sepal Width, Petal Length, Petal Width) |
| Source     | Built-in via `sklearn.datasets.load_iris()`  |

---

## ⚙️ Tech Stack

| Tool            | Purpose                        |
|-----------------|--------------------------------|
| Python 3.8+     | Programming language           |
| scikit-learn    | KNN model, scaling, metrics    |
| pandas          | Data handling                  |
| numpy           | Numerical operations           |
| matplotlib      | Plotting & dashboard           |
| seaborn         | Heatmap visualization          |

---

## 🔁 Pipeline (IPO Framework)

```
INPUT              →        PROCESS           →       OUTPUT
─────────────────────────────────────────────────────────────
Iris Dataset               Train-Test Split          Confusion Matrix
Feature Scaling            KNN Algorithm             F1 Score
                           Elbow Method              Dashboard PNG
```

### Steps Implemented

1. **Load & Explore** — Load dataset, check shape, class distribution, statistics
2. **Feature Scaling** — `StandardScaler` → mean=0, variance=1
3. **Train/Test Split** — 80% train / 20% test with shuffle + stratify
4. **Find Optimal K** — Elbow method tested K=1 to K=20
5. **Train Model** — `KNeighborsClassifier(n_neighbors=optimal_k)`
6. **Evaluate** — Accuracy, F1 Score, Confusion Matrix, Classification Report
7. **Visualize** — 9-panel dashboard saved as PNG
8. **Live Predictions** — 3 custom flower samples classified in real time

---

## 📊 Results

| Metric         | Score     |
|----------------|-----------|
| Accuracy       | **96.67%** |
| F1 Score       | **0.9666** |
| Optimal K      | 1          |
| Train Samples  | 120        |
| Test Samples   | 30         |

### Per-Class Performance

| Class       | Precision | Recall | F1 Score |
|-------------|-----------|--------|----------|
| Setosa      | 1.00      | 1.00   | 1.00     |
| Versicolor  | 0.91      | 1.00   | 0.95     |
| Virginica   | 1.00      | 0.90   | 0.95     |

---

## 🚀 How to Run

### 1. Clone / Download the project folder

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### 4. Run the project
```bash
python iris_knn_project.py
```

### 5. View the dashboard
Open `iris_knn_dashboard.png` in your project folder.

---

## 📈 Dashboard Panels

The output dashboard (`iris_knn_dashboard.png`) contains 9 panels:

1. Class Distribution Bar Chart
2. Feature Correlation Heatmap
3. Petal Dimensions Scatter Plot
4. Elbow Method — Error Rate vs K
5. F1 Score vs K
6. Confusion Matrix
7. Per-Class F1 Score
8. Precision vs Recall
9. Summary Results Card

---

## 💡 Key Concepts Used

- **Supervised Learning** — Model learns from labeled data
- **StandardScaler** — Removes bias caused by different feature scales
- **KNN Algorithm** — Classifies based on majority vote of K nearest neighbors
- **Elbow Method** — Finds the optimal K with lowest error rate
- **Confusion Matrix** — Visual breakdown of TP, FP, FN, TN
- **F1 Score** — Harmonic mean of Precision and Recall 