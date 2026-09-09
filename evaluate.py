from pathlib import Path
import json

import numpy as np
import keras
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)


# ============================================================
# TomatoDoc - Model Evaluation
# ============================================================

MODEL_PATH = Path(
    "models/tomatodoc_best.keras"
)

TEST_DIR = Path(
    "processed_dataset/test"
)

MODEL_DIR = Path("models")

IMAGE_SIZE = (
    224,
    224
)

BATCH_SIZE = 32


# ============================================================
# Class names
# ============================================================

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]


DISPLAY_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus",
]


# ============================================================
# Check files
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_DIR}"
    )


# ============================================================
# Load model
# ============================================================

print("=" * 70)
print("TomatoDoc Model Evaluation")
print("=" * 70)

print()
print("Loading model...")

model = keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# Load test dataset
# ============================================================

print()
print("=" * 70)
print("Loading test dataset...")
print("=" * 70)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# Evaluate loss and accuracy
# ============================================================

print()
print("=" * 70)
print("Evaluating...")
print("=" * 70)

loss, accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print()
print(
    f"Test Loss     : {loss:.4f}"
)

print(
    f"Test Accuracy : {accuracy:.4f}"
)

print(
    f"Test Accuracy : {accuracy * 100:.2f}%"
)


# ============================================================
# Generate predictions
# ============================================================

print()
print("Generating predictions...")

y_true = []
y_pred = []


for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# Accuracy
# ============================================================

final_accuracy = accuracy_score(
    y_true,
    y_pred
)


print()
print(
    f"Calculated Accuracy: "
    f"{final_accuracy * 100:.2f}%"
)


# ============================================================
# Classification Report
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=DISPLAY_NAMES,
    digits=4,
    zero_division=0
)

print(report)


# Save report
with open(
    MODEL_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# Confusion Matrix
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)


# ============================================================
# Save confusion matrix image
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=DISPLAY_NAMES
)

fig, ax = plt.subplots(
    figsize=(12, 10)
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title(
    "TomatoDoc MobileNetV2 Confusion Matrix"
)

plt.tight_layout()

confusion_path = (
    MODEL_DIR / "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Final
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print()
print(
    "Classification report:",
    MODEL_DIR / "classification_report.txt"
)

print(
    "Confusion matrix:",
    confusion_path
)