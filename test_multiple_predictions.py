import os
import json
import random
import numpy as np
from tensorflow import keras


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/tomatodoc_best.keras"
CLASS_FILE = "models/class_names.json"
TEST_FOLDER = "processed_dataset/test"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading TomatoDoc AI model...\n")

model = keras.models.load_model(MODEL_PATH)

print("✅ Model loaded successfully!\n")


# ==========================================
# LOAD CLASS NAMES
# ==========================================

with open(CLASS_FILE, "r") as file:
    class_data = json.load(file)

class_names = class_data["classes"]
display_names = class_data["display_names"]


# ==========================================
# TEST ONE RANDOM IMAGE FROM EACH CLASS
# ==========================================

correct_predictions = 0
total_predictions = 0

print("=" * 65)
print("TOMATODOC MULTIPLE DISEASE TEST")
print("=" * 65)


for class_index, class_name in enumerate(class_names):

    class_folder = os.path.join(TEST_FOLDER, class_name)

    images = [
        file for file in os.listdir(class_folder)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if not images:
        print(f"\n⚠️ No images found for {class_name}")
        continue

    # Select one random image
    selected_image = random.choice(images)

    image_path = os.path.join(
        class_folder,
        selected_image
    )


    # ==========================================
    # LOAD IMAGE
    # ==========================================

    image = keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    image_array = keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # IMPORTANT:
    # Do NOT divide by 255
    # Model already handles preprocessing


    # ==========================================
    # PREDICT
    # ==========================================

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])

    confidence = (
        predictions[0][predicted_index] * 100
    )

    actual_disease = display_names[class_index]
    predicted_disease = display_names[predicted_index]


    # ==========================================
    # SHOW RESULT
    # ==========================================

    total_predictions += 1

    print("\n" + "-" * 65)

    print(f"Actual:    {actual_disease}")
    print(f"Predicted: {predicted_disease}")
    print(f"Confidence: {confidence:.2f}%")

    if predicted_index == class_index:

        print("Result:    ✅ CORRECT")

        correct_predictions += 1

    else:

        print("Result:    ❌ INCORRECT")


# ==========================================
# FINAL RESULT
# ==========================================

accuracy = (
    correct_predictions / total_predictions
) * 100


print("\n" + "=" * 65)
print("FINAL TEST RESULT")
print("=" * 65)

print(f"Correct Predictions: {correct_predictions}")
print(f"Total Tested:        {total_predictions}")
print(f"Test Accuracy:       {accuracy:.2f}%")

print("=" * 65)