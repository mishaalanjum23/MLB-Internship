import zipfile

# Save the original standard library methods
_original_open = zipfile.ZipFile.open
_original_getinfo = zipfile.ZipFile.getinfo

# Define patched versions that replace backslashes with forward slashes
def patched_open(self, name, *args, **kwargs):
    if isinstance(name, str):
        name = name.replace('\\', '/')
    return _original_open(self, name, *args, **kwargs)

def patched_getinfo(self, name):
    if isinstance(name, str):
        name = name.replace('\\', '/')
    return _original_getinfo(self, name)

# Apply the patch globally
zipfile.ZipFile.open = patched_open
zipfile.ZipFile.getinfo = patched_getinfo

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt

# Load dataset
dataset, info = tfds.load(
    "cats_vs_dogs",
    with_info=True,
    as_supervised=True
)
dataset = dataset["train"]

# Preprocessing
def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = preprocess_input(image)
    return image, label

dataset = dataset.map(preprocess)

# Split into training and validation sets
dataset_size = info.splits["train"].num_examples
train_size = int(0.8 * dataset_size)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Shuffle and batch
train_dataset = train_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# using MobileNetV2 as base model
base_model = MobileNetV2(weights = "imagenet", include_top = False, input_shape = (224, 224, 3))

# freeze layers
base_model.trainable = False

model = Sequential([base_model,
                  GlobalAveragePooling2D(),
                  Dense(128, activation="relu"),
                  Dense(2, activation="softmax")
                ])

# train model
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
record = model.fit(train_dataset, epochs = 5, validation_data = val_dataset)

# evaluate
val_loss, val_acc = model.evaluate(val_dataset)
print("Validation Accuracy:", val_acc)
print("Validation Loss:", val_loss)

# Display sample predictions
class_names = info.features["label"].names
for images, labels in val_dataset.take(1):
    predictions = model.predict(images)
    predicted_labels = tf.argmax(predictions, axis=1)

    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow((images[i] + 1) / 2)
        plt.title(
            f"Pred: {class_names[predicted_labels[i]]}\nTrue: {class_names[labels[i]]}"
        )
        plt.axis("off")
    plt.show()

# Plot accuracy
plt.figure(figsize=(6,4))
plt.plot(record.history["accuracy"], label="Training Accuracy")
plt.plot(record.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.show()

# Plot loss
plt.figure(figsize=(6,4))
plt.plot(record.history["loss"], label="Training Loss")
plt.plot(record.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.show()