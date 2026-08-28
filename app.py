import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --------------------------------------------------
# Dataset Paths
# --------------------------------------------------
train_path = "dataset/Training"
test_path = "dataset/Testing"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# --------------------------------------------------
# Data Augmentation
# --------------------------------------------------
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(
    rescale=1.0 / 255
)

# --------------------------------------------------
# Training Data
# --------------------------------------------------
train_data = train_gen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

# --------------------------------------------------
# Testing Data
# --------------------------------------------------
test_data = test_gen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Print class mapping
print("Class Mapping:")
print(train_data.class_indices)

# --------------------------------------------------
# Load Pretrained MobileNetV2
# --------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# --------------------------------------------------
# Build Model
# --------------------------------------------------
x = base_model.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.5)(x)

output = tf.keras.layers.Dense(
    4,
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs=base_model.input,
    outputs=output
)

# --------------------------------------------------
# Compile Model
# --------------------------------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# Train Model
# --------------------------------------------------
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=25
)

# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------
loss, accuracy = model.evaluate(test_data)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --------------------------------------------------
# Save Model
# --------------------------------------------------
model.save("brain_tumor_model.h5")

print(
    "✅ Model training complete and saved as "
    "brain_tumor_model.h5"
)
