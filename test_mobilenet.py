import keras

print("Keras version:", keras.__version__)
print("Backend:", keras.backend.backend())

print("\nLoading MobileNetV2...")

model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

print("\nMobileNetV2 loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Number of layers:", len(model.layers))