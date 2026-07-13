import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
import matplotlib.pyplot as plt

# Load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Explore dataset
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# see actual image and label
plt.imshow(train_images[0])
plt.savefig("Training Image.png")
plt.show()
print(train_labels[0])

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build ANN
model = Sequential()
model.add(Input(shape = (28, 28)))
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dense(10, activation = "softmax"))
model.summary()

# Compile model
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# Training model
record = model.fit(train_images, train_labels, epochs = 5, validation_split = 0.2)

# Training and validation accuracy
print("Final Training Accuracy:", record.history["accuracy"][-1])
print("Final Validation Accuracy:", record.history["val_accuracy"][-1])

# Evaluate
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# Predict
predictions = model.predict(test_images)
for i in range (5):
 print("\nImage", i + 1)
 print("Predicted Label:", predictions[i].argmax())
 print("Actual Label:", test_labels[i])
 