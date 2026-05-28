# ============================================================
# PROJECT 2: Data Classification Using AI
# DecodeLabs Industrial Training Kit - Batch 2026
# Algorithm: K-Nearest Neighbors (KNN)
# Dataset: Iris Benchmark (150 samples, 3 classes, 4 features)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, accuracy_score, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  PROJECT 2: DATA CLASSIFICATION USING AI")
print("  DecodeLabs | KNN on Iris Dataset")
print("=" * 60)

# STEP 1 — LOAD & UNDERSTAND THE DATASET
print("\n📦 STEP 1: Loading the Iris Dataset...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
class_names = iris.target_names

print(f"\n  Shape       : {X.shape} → 150 samples × 4 features")
print(f"  Classes     : {list(class_names)}")
print(f"  Class Dist  : {dict(zip(class_names, np.bincount(y)))}")
print(f"\n  First 5 rows:\n{X.head()}")
print(f"\n  Statistical Summary:\n{X.describe().round(2)}")

# STEP 2 — FEATURE SCALING
print("\n⚖️  STEP 2: Feature Scaling with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  Before scaling — mean: {X.mean().values.round(2)}")
print(f"  After  scaling — mean: {X_scaled.mean(axis=0).round(4)}  (≈ 0)")
print(f"  After  scaling — std : {X_scaled.std(axis=0).round(4)}   (≈ 1)")

# STEP 3 — TRAIN/TEST SPLIT
print("\n✂️  STEP 3: Splitting Data (80% Train / 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")
print(f"  Stratified split — class balance preserved ✓")

# STEP 4 — FIND OPTIMAL K
print("\n🔍 STEP 4: Finding Optimal K (Elbow Method)...")
k_range = range(1, 21)
error_rates = []
f1_scores   = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))
    f1_scores.append(f1_score(y_test, preds, average='weighted'))

optimal_k = k_range[np.argmin(error_rates)]
print(f"  Optimal K = {optimal_k}  (lowest error rate: {min(error_rates):.4f})")

# STEP 5 — TRAIN FINAL MODEL
print(f"\n🤖 STEP 5: Training KNN Model with K={optimal_k}...")
model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)
print("  Model trained ✓")

# STEP 6 — PREDICT & EVALUATE
print("\n📊 STEP 6: Predictions & Evaluation...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred, average='weighted')
cm       = confusion_matrix(y_test, y_pred)

print(f"\n  ✅ Accuracy  : {accuracy * 100:.2f}%")
print(f"  ✅ F1 Score  : {f1:.4f}  (weighted)")
print(f"\n  Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))
print(f"  Confusion Matrix:\n{cm}")

# STEP 7 — VISUALISATION DASHBOARD
print("\n🎨 STEP 7: Generating Visualisation Dashboard...")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#f0f4f8')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

BLUE   = '#1a3a5c'
ORANGE = '#e86c2b'
GREEN  = '#2d7a4f'
LIGHT  = '#eaf2fb'

ax1 = fig.add_subplot(gs[0, 0])
counts = np.bincount(iris.target)
bars = ax1.bar(class_names, counts, color=[BLUE, ORANGE, GREEN], edgecolor='white', linewidth=1.5)
for bar, c in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(c), ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.set_title('Dataset: Class Distribution', fontweight='bold', color=BLUE)
ax1.set_ylabel('Count')
ax1.set_facecolor(LIGHT)
ax1.spines[['top','right']].set_visible(False)

ax2 = fig.add_subplot(gs[0, 1])
corr = pd.DataFrame(X_scaled, columns=iris.feature_names).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax2.set_title('Feature Correlation Heatmap', fontweight='bold', color=BLUE)
ax2.tick_params(axis='x', rotation=30, labelsize=7)
ax2.tick_params(axis='y', rotation=0, labelsize=7)

ax3 = fig.add_subplot(gs[0, 2])
colors = [BLUE, ORANGE, GREEN]
for i, (cls, col) in enumerate(zip(class_names, colors)):
    mask = iris.target == i
    ax3.scatter(iris.data[mask, 2], iris.data[mask, 3],
                c=col, label=cls, alpha=0.75, edgecolors='white', s=60)
ax3.set_xlabel('Petal Length (cm)')
ax3.set_ylabel('Petal Width (cm)')
ax3.set_title('Feature Scatter: Petal Dims', fontweight='bold', color=BLUE)
ax3.legend(fontsize=8)
ax3.set_facecolor(LIGHT)
ax3.spines[['top','right']].set_visible(False)

ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(list(k_range), error_rates, 'o-', color=BLUE, linewidth=2, markersize=6)
ax4.scatter([optimal_k], [min(error_rates)], color=ORANGE, s=160, zorder=5,
            label=f'Optimal K={optimal_k}', edgecolors='white', linewidth=2)
ax4.set_xlabel('K Value')
ax4.set_ylabel('Error Rate')
ax4.set_title('Elbow Method: Finding Optimal K', fontweight='bold', color=BLUE)
ax4.legend(fontsize=9)
ax4.set_facecolor(LIGHT)
ax4.spines[['top','right']].set_visible(False)

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(list(k_range), f1_scores, 's-', color=GREEN, linewidth=2, markersize=6)
ax5.scatter([optimal_k], [f1_scores[optimal_k - 1]], color=ORANGE, s=160, zorder=5,
            label=f'Best F1={f1_scores[optimal_k-1]:.3f}', edgecolors='white', linewidth=2)
ax5.set_xlabel('K Value')
ax5.set_ylabel('F1 Score (weighted)')
ax5.set_title('F1 Score vs K', fontweight='bold', color=BLUE)
ax5.legend(fontsize=9)
ax5.set_facecolor(LIGHT)
ax5.spines[['top','right']].set_visible(False)

ax6 = fig.add_subplot(gs[1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax6, colorbar=False, cmap='Blues')
ax6.set_title('Confusion Matrix', fontweight='bold', color=BLUE)
ax6.tick_params(axis='x', rotation=20)

ax7 = fig.add_subplot(gs[2, 0])
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
cls_f1 = [report[c]['f1-score'] for c in class_names]
bars7  = ax7.bar(class_names, cls_f1, color=[BLUE, ORANGE, GREEN], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars7, cls_f1):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax7.set_ylim(0, 1.15)
ax7.set_ylabel('F1 Score')
ax7.set_title('Per-Class F1 Score', fontweight='bold', color=BLUE)
ax7.set_facecolor(LIGHT)
ax7.spines[['top','right']].set_visible(False)

ax8 = fig.add_subplot(gs[2, 1])
prec = [report[c]['precision'] for c in class_names]
rec  = [report[c]['recall']    for c in class_names]
x    = np.arange(len(class_names))
w    = 0.35
ax8.bar(x - w/2, prec, w, label='Precision', color=BLUE,   edgecolor='white')
ax8.bar(x + w/2, rec,  w, label='Recall',    color=ORANGE, edgecolor='white')
ax8.set_xticks(x)
ax8.set_xticklabels(class_names)
ax8.set_ylim(0, 1.2)
ax8.set_ylabel('Score')
ax8.set_title('Precision vs Recall', fontweight='bold', color=BLUE)
ax8.legend(fontsize=9)
ax8.set_facecolor(LIGHT)
ax8.spines[['top','right']].set_visible(False)

ax9 = fig.add_subplot(gs[2, 2])
ax9.set_facecolor(BLUE)
ax9.set_xlim(0, 1); ax9.set_ylim(0, 1)
ax9.axis('off')
metrics_text = [
    ("PROJECT 2 RESULTS", 0.88, 16, 'white', 'bold'),
    ("DecodeLabs · KNN Classifier", 0.76, 9, '#aac8e8', 'normal'),
    (f"Accuracy:   {accuracy*100:.2f}%", 0.60, 13, ORANGE, 'bold'),
    (f"F1 Score:   {f1:.4f}", 0.47, 13, ORANGE, 'bold'),
    (f"Optimal K:  {optimal_k}", 0.34, 13, 'white', 'bold'),
    (f"Train/Test: 80% / 20%", 0.22, 11, '#aac8e8', 'normal'),
    ("Algorithm:  KNN + StandardScaler", 0.11, 9, '#aac8e8', 'normal'),
]
for txt, y_pos, fs, col, fw in metrics_text:
    ax9.text(0.5, y_pos, txt, ha='center', va='center',
             fontsize=fs, color=col, fontweight=fw,
             transform=ax9.transAxes)

fig.suptitle('PROJECT 2: DATA CLASSIFICATION USING AI  |  DecodeLabs Batch 2026',
             fontsize=14, fontweight='bold', color=BLUE, y=0.98)

plt.savefig('iris_knn_dashboard.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  Dashboard saved → iris_knn_dashboard.png ✓")

# STEP 8 — LIVE PREDICTION DEMO
print("\n🌸 STEP 8: Live Prediction Demo")
demo_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.7, 3.0, 5.2, 2.3],
])
demo_scaled = scaler.transform(demo_samples)
demo_preds  = model.predict(demo_scaled)
demo_proba  = model.predict_proba(demo_scaled)

print(f"\n  {'Sample':<8} {'Sepal L':>8} {'Sepal W':>8} {'Petal L':>8} {'Petal W':>8}  {'Predicted':<12} {'Confidence'}")
print("  " + "-" * 72)
for i, (sample, pred, proba) in enumerate(zip(demo_samples, demo_preds, demo_proba)):
    conf = max(proba) * 100
    print(f"  Sample {i+1}  {sample[0]:>8.1f} {sample[1]:>8.1f} {sample[2]:>8.1f} {sample[3]:>8.1f}  {class_names[pred]:<12} {conf:.1f}%")

print("\n" + "=" * 60)
print("  ✅ PROJECT 2 COMPLETE!")
print(f"  Final Accuracy : {accuracy * 100:.2f}%")
print(f"  Final F1 Score : {f1:.4f}")
print("  Dashboard saved as iris_knn_dashboard.png")
print("=" * 60)