import os
import json
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/tomatodoc_best.keras"
CLASS_FILE = "models/class_names.json"
TEST_FOLDER = "processed_dataset/test"

IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading TomatoDoc AI model...")

model = keras.models.load_model(MODEL_PATH)

print("✅ Model loaded successfully!")


# ==========================================
# LOAD CLASS NAMES
# ==========================================

with open(CLASS_FILE, "r") as file:
    class_data = json.load(file)

class_names = class_data["classes"]
display_names = class_data["display_names"]

print("\nClasses loaded:")
print(f"Total classes: {len(class_names)}")


# ==========================================
# VARIABLES
# ==========================================

true_labels = []
predicted_labels = []

total_images = 0
correct_predictions = 0


# ==========================================
# TEST EVERY IMAGE
# ==========================================

print("\n" + "=" * 60)
print("STARTING FULL TOMATODOC MODEL TEST")
print("=" * 60)


for class_index, class_name in enumerate(class_names):

    class_folder = os.path.join(
        TEST_FOLDER,
        class_name
    )

    print(f"\nTesting: {display_names[class_index]}")

    images = [
        file for file in os.listdir(class_folder)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ]

    class_total = 0
    class_correct = 0


    for filename in images:

        image_path = os.path.join(
            class_folder,
            filename
        )


        # ==========================================
        # LOAD IMAGE
        # ==========================================

        try:

            image = keras.utils.load_img(
                image_path,
                target_size=IMAGE_SIZE
            )

            image_array = keras.utils.img_to_array(image)

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )


            # IMPORTANT:
            # Do NOT divide by 255
            # The model already has preprocessing


            # ==========================================
            # PREDICT
            # ==========================================

            predictions = model.predict(
                image_array,
                verbose=0
            )

            predicted_index = int(
                np.argmax(predictions[0])
            )


            # ==========================================
            # SAVE RESULTS
            # ==========================================

            true_labels.append(class_index)

            predicted_labels.append(predicted_index)

            total_images += 1
            class_total += 1


            if predicted_index == class_index:

                correct_predictions += 1
                class_correct += 1


        except Exception as error:

            print(
                f"\n⚠️ Error reading {filename}: {error}"
            )


    # ==========================================
    # CLASS RESULT
    # ==========================================

    if class_total > 0:

        class_accuracy = (
            class_correct / class_total
        ) * 100

        print(
            f"Result: {class_correct}/{class_total} "
            f"correct ({class_accuracy:.2f}%)"
        )


# ==========================================
# FINAL ACCURACY
# ==========================================

accuracy = (
    correct_predictions / total_images
) * 100


print("\n" + "=" * 60)
print("FINAL TOMATODOC MODEL RESULT")
print("=" * 60)

print(f"Total Images Tested: {total_images}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {total_images - correct_predictions}")
print(f"\nOverall Accuracy: {accuracy:.2f}%")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=display_names,
    digits=4
)

print(report)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

matrix = confusion_matrix(
    true_labels,
    predicted_labels
)

print(matrix)


print("\n✅ FULL TEST COMPLETE!")