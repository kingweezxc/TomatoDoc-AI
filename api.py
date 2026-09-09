import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# TomatoDoc FastAPI Backend
# ============================================================

app = FastAPI(
    title="TomatoDoc API",
    description="Tomato leaf disease detection using MobileNetV2",
    version="1.0.0"
)


# ============================================================
# CORS
# Allows the TomatoDoc PWA/frontend to communicate with the API
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Load Model
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "tomatodoc_mobilenetv2.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

print("Loading TomatoDoc model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

print("Loading class names...")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names_data = json.load(f)

# Handle TomatoDoc class_names.json
if isinstance(class_names_data, dict):

    if "display_names" in class_names_data:
        class_names = class_names_data["display_names"]

    elif "classes" in class_names_data:
        class_names = class_names_data["classes"]

    else:
        class_names = list(class_names_data.values())

else:
    class_names = class_names_data


# Make sure the number of classes matches the model
model_class_count = model.output_shape[-1]

if len(class_names) != model_class_count:
    raise ValueError(
        f"Class count mismatch: model outputs "
        f"{model_class_count} classes, but "
        f"class_names.json contains {len(class_names)} names."
    )

print("Classes:", class_names)

print(
    "Available GPUs:",
    tf.config.list_physical_devices("GPU")
)

# ============================================================
# Basic API test
# ============================================================

@app.get("/")
def root():
    return {
        "message": "TomatoDoc API is running",
        "model": "MobileNetV2",
        "classes": class_names,
        "gpu": [
            device.name
            for device in tf.config.list_physical_devices("GPU")
        ]
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "gpu_available": len(tf.config.list_physical_devices("GPU")) > 0
    }


# ============================================================
# Disease Prediction
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check that an image was uploaded
    if not file.content_type or not file.content_type.startswith("image/"):
        return {
            "success": False,
            "error": "Please upload an image file."
        }

    try:
        # Read uploaded image
        image_data = await file.read()

        # Open image
        image = Image.open(
            __import__("io").BytesIO(image_data)
        )

        # Convert to RGB
        image = image.convert("RGB")

        # Resize to MobileNetV2 input size
        image = image.resize((224, 224))

        # Convert image to numpy array
        image_array = np.array(image, dtype=np.float32)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # ----------------------------------------------------
        # IMPORTANT:
        # The trained TomatoDoc model already contains its
        # MobileNetV2 preprocessing/rescaling layer.
        # Therefore we DO NOT call preprocess_input here.
        # ----------------------------------------------------

        # Make prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        # Find highest probability
        predicted_index = int(np.argmax(predictions))

        predicted_class = class_names[predicted_index]

        confidence = float(predictions[predicted_index] * 100)

        # Get top 3 predictions
        top_indices = np.argsort(predictions)[::-1][:3]

        top_predictions = []

        for index in top_indices:
            top_predictions.append({
                "disease": class_names[int(index)],
                "confidence": round(
                    float(predictions[index] * 100),
                    2
                )
            })

        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "top_predictions": top_predictions
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }