from tensorflow.keras.models import load_model
m = load_model("brain_tumor_model.h5")   # or brain_tumor_4class.h5 if you later use that
print("Input shape:", m.input_shape)
print("Output shape:", m.output_shape)
