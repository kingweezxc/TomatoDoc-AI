import os
import json
import numpy as np
from tensorflow import keras


# ==========================================
# 1. LOAD THE TRAINED MODEL
# ==========================================

MODEL_PATH = "models/tomatodoc_best.keras"

print("\nLoading TomatoDoc AI model...")
model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# 2. LOAD CLASS NAMES
# ==========================================

with open("models/class_names.json", "r") as file:
    class_data = json.load(file)

class_names = class_data["classes"]
display_names = class_data["display_names"]


# ==========================================
# 3. FIND ONE TEST IMAGE
# ==========================================

TEST_FOLDER = "processed_dataset/test"

image_extensions = (".jpg", ".jpeg", ".png")

test_image = None

for root, folders, files in os.walk(TEST_FOLDER):
    for filename in files:
        if filename.lower().endswith(image_extensions):
            test_image = os.path.join(root, filename)
            break

    if test_image:
        break


if not test_image:
    print("\nERROR: No test image found!")
    exit()


# Get actual class from parent folder
actual_class = os.path.basename(os.path.dirname(test_image))

actual_index = class_names.index(actual_class)

actual_disease = display_names[actual_index]


print("\nTesting image:")
print(test_image)

print("\nActual Disease:")
print(actual_disease)


# ==========================================
# 4. PREPARE IMAGE
# ==========================================

image = keras.utils.load_img(
    test_image,
    target_size=(224, 224)
)

image_array = keras.utils.img_to_array(image)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# IMPORTANT:
# Do NOT divide by 255.
# The saved model already contains MobileNetV2 preprocessing.


# ==========================================
# 5. MAKE PREDICTION
# ==========================================

predictions = model.predict(
    image_array,
    verbose=0
)

predicted_index = np.argmax(predictions[0])

confidence = predictions[0][predicted_index] * 100

predicted_disease = display_names[predicted_index]


# ==========================================
# 6. SHOW RESULT
# ==========================================

print("\n" + "=" * 55)
print("TOMATODOC AI DIAGNOSIS RESULT")
print("=" * 55)

print(f"\nActual Disease: {actual_disease}")
print(f"AI Prediction: {predicted_disease}")
print(f"Confidence: {confidence:.2f}%")


# ==========================================
# 7. CHECK RESULT
# ==========================================

if predicted_index == actual_index:
    print("\n✅ RESULT: CORRECT PREDICTION")
else:
    print("\n❌ RESULT: INCORRECT PREDICTION")


# ==========================================
# 8. SHOW TOP 3
# ==========================================

top_3 = np.argsort(predictions[0])[-3:][::-1]

print("\nTOP 3 PREDICTIONS:")

for index in top_3:
    score = predictions[0][index] * 100
    print(
        f"{display_names[index]}: "
        f"{score:.2f}%"
    )

print("\n" + "=" * 55)