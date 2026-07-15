import tensorflow as tf
import tensorflow_datasets as tfds

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
    image = image / 255.0
    return image, label

# Apply preprocessing
dataset = dataset.map(preprocess)

# Split into training and validation sets
dataset_size = info.splits["train"].num_examples
train_size = int(0.8 * dataset_size)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

print("Training samples:", train_size)
print("Validation samples:", dataset_size - train_size)