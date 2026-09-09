from tensorflow import keras
import os

models = [
    "models/tomatodoc_mobilenetv2.keras",
    "models/tomatodoc_best.keras"
]

for model_path in models:
    print("\n" + "=" * 60)
    print(f"CHECKING: {model_path}")
    print("=" * 60)

    if not os.path.exists(model_path):
        print("❌ File not found!")
        continue

    try:
        model = keras.models.load_model(model_path)

        print("✅ Model loaded successfully!")
        print("Model name:", model.name)
        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)
        print("Total layers:", len(model.layers))

    except Exception as e:
        print("❌ Error loading model:")
        print(e)