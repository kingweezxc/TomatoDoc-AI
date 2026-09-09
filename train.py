import json
from pathlib import Path

import keras
from keras import layers


# ============================================================
# TomatoDoc - MobileNetV2 Training
# ============================================================

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATASET_DIR = Path("processed_dataset")

TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

IMAGE_SIZE = (
    224,
    224
)

BATCH_SIZE = 32

NUM_CLASSES = 10

INITIAL_EPOCHS = 20

FINE_TUNE_EPOCHS = 20

SEED = 42


# ============================================================
# TomatoDoc Classes
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
# Print configuration
# ============================================================

print("=" * 70)
print("TomatoDoc - MobileNetV2 Training")
print("=" * 70)

print()
print("Keras version:", keras.__version__)
print("Backend:", keras.backend.backend())

print()
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Number of classes:", NUM_CLASSES)

print()
print("Classes:")

for index, name in enumerate(DISPLAY_NAMES):
    print(
        f"  {index}: {name}"
    )


# ============================================================
# Check dataset
# ============================================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}"
    )

if not VALIDATION_DIR.exists():
    raise FileNotFoundError(
        f"Validation directory not found: {VALIDATION_DIR}"
    )


# ============================================================
# Load training dataset
# ============================================================

print()
print("=" * 70)
print("Loading training dataset...")
print("=" * 70)

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)


# ============================================================
# Load validation dataset
# ============================================================

print()
print("=" * 70)
print("Loading validation dataset...")
print("=" * 70)

validation_ds = keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# Performance optimization
# ============================================================

AUTOTUNE = keras.backend.backend()

# Keras/TensorFlow will handle dataset execution.
# Prefetching is applied through the TensorFlow dataset API.

try:
    import tensorflow as tf

    train_ds = train_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    validation_ds = validation_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

except Exception:
    print(
        "TensorFlow prefetch optimization skipped."
    )


# ============================================================
# Data augmentation
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),

        layers.RandomRotation(0.20),

        layers.RandomZoom(
            height_factor=0.20,
            width_factor=0.20
        ),

        layers.RandomTranslation(
            height_factor=0.12,
            width_factor=0.12
        ),

        layers.RandomContrast(0.20),
    ],
    name="data_augmentation"
)

# ============================================================
# MobileNetV2 Base Model
# ============================================================

print()
print("=" * 70)
print("Loading MobileNetV2...")
print("=" * 70)

base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(
        224,
        224,
        3
    ),
)


# Freeze pretrained layers
base_model.trainable = False


# ============================================================
# Build TomatoDoc Model
# ============================================================

inputs = keras.Input(
    shape=(
        224,
        224,
        3
    ),
    name="tomato_leaf_image"
)


# Data augmentation
x = data_augmentation(inputs)


# MobileNetV2 preprocessing
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1.0,
    name="mobilenet_preprocessing"
)(x)


# Feature extraction
x = base_model(
    x,
    training=False
)


# Reduce feature dimensions
x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

x = layers.Dense(
    256,
    activation="relu",
    name="dense_features"
)(x)

x = layers.Dropout(
    0.40,
    name="dropout"
)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="disease_prediction"
)(x)


model = keras.Model(
    inputs,
    outputs,
    name="TomatoDoc_MobileNetV2"
)


# ============================================================
# Display model
# ============================================================

print()
model.summary()


# ============================================================
# Compile - Stage 1
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ],
)


# ============================================================
# Callbacks
# ============================================================

stage1_checkpoint_path = (
    MODEL_DIR / "tomatodoc_stage1_best.keras"
)

stage2_checkpoint_path = (
    MODEL_DIR / "tomatodoc_stage2_best.keras"
)
callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath=str(
            stage1_checkpoint_path
        ),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),

    keras.callbacks.CSVLogger(
        str(
            MODEL_DIR / "training_log.csv"
        )
    ),
]


# ============================================================
# Stage 1 - Train Classification Head
# ============================================================

print()
print("=" * 70)
print("STAGE 1: Training classification head")
print("=" * 70)

history_initial = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
)


# ============================================================
# Stage 2 - Fine Tuning
# ============================================================

print()
print("=" * 70)
print("STAGE 2: Fine-tuning MobileNetV2")
print("=" * 70)


# Unfreeze MobileNetV2
base_model.trainable = True


# Freeze the first part of MobileNetV2.
# This keeps low-level visual features stable.
fine_tune_from = 60

for layer in base_model.layers[
    :fine_tune_from
]:
    layer.trainable = False


# Recompile after changing trainable layers
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ],
)


# Fine-tuning callbacks
fine_tune_callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath=str(
            stage2_checkpoint_path
        ),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-8,
        verbose=1,
    ),

   keras.callbacks.CSVLogger(
    str(
        MODEL_DIR / "fine_tuning_log.csv"
    ),
    append=True,
),
]


history_fine = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=fine_tune_callbacks,
)


# ============================================================
# Save final model
# ============================================================

final_model_path = (
    MODEL_DIR / "tomatodoc_mobilenetv2.keras"
)

model.save(
    final_model_path
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"\nStage 1 best model: {stage1_checkpoint_path}")
print(f"Stage 2 best model: {stage2_checkpoint_path}")
print(f"Final model: {final_model_path}")
print(f"Class mapping: {MODEL_DIR / 'class_names.json'}")

# ============================================================
# Save class names
# ============================================================

class_mapping = {
    "classes": CLASS_NAMES,
    "display_names": DISPLAY_NAMES,
}


with open(
    MODEL_DIR / "class_names.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_mapping,
        file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# Final output
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print(
    "Best model:",
    
)

print(
    "Final model:",
    final_model_path
)

print(
    "Class mapping:",
    MODEL_DIR / "class_names.json"
)

print()
print("TomatoDoc MobileNetV2 training finished!")