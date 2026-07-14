import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense 
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# display sample images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i], cmap="gray")
    plt.title(class_names[test_labels[i]])
    plt.axis("off")
plt.tight_layout()
plt.show()

# Normalizing data
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Build a CNN model
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dense(10, activation = "softmax")) 

# training model
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
record = model.fit(train_images, train_labels, epochs = 5, batch_size = 32, validation_split = 0.2)

# Evaluate the model
print("Training Accuracy:", record.history["accuracy"][-1])
print("Validation Accuracy:", record.history["val_accuracy"][-1])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

# Predict on test data
predictions = model.predict(test_images) 

# Display predictions
for i in range (5):
 predicted_label = predictions[i].argmax()
 print("\nImage", i + 1)
 print("Predicted Label:", class_names[predicted_label]) 
 print("Actual Label:", class_names[test_labels[i]])

# Plots the training and validation accuracy curves.
plt.plot(record.history["accuracy"], label = "Training accuracy")
plt.plot(record.history["val_accuracy"], label = "Validation accuracy")
plt.title("Training and Validation Accuracy") 
plt.xlabel("Epoch")
plt.ylabel("Accuracy") 
plt.legend()
plt.show()

# Display a confusion matrix.
predicted_labels = predictions.argmax(axis=1)
ConfusionMatrixDisplay.from_predictions(test_labels, predicted_labels)
plt.show()

# Show 10 correctly classified and 10 incorrectly classified images with their predicted labels.
correct_images = []
incorrect_images = []

for i in range(len(test_images)): 
   if predicted_labels[i] == test_labels[i]:
      correct_images.append(i)
   else:
      incorrect_images.append(i)

# Correctly classified images
plt.figure(figsize=(10, 5))
for i in range(10):
    index = correct_images[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[index].squeeze(), cmap="gray")
    plt.title(f"P: {class_names[predicted_labels[index]]}\nA: {class_names[test_labels[index]]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# incorrectly classified images
plt.figure(figsize=(10, 5))
for i in range(10):
    index = incorrect_images[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[index].squeeze(), cmap="gray")
    plt.title(f"P: {class_names[predicted_labels[index]]}\nA: {class_names[test_labels[index]]}")
    plt.axis("off")

plt.tight_layout()
plt.show()