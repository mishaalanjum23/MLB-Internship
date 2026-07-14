import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense 
import matplotlib.pyplot as plt

# practice 1: load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# visualize 10 images and their labels
for i in range(10):
    plt.imshow(train_images[i])
    plt.title(train_labels[i])
    plt.show()

# Normalizing data
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Practice 2: Build a CNN model
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

# Practice 3: Evaluate the model
print("Training Accuracy:", record.history["accuracy"][-1])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

predictions = model.predict(test_images)
for i in range (5):
 print("\nImage", i + 1)
 print("Predicted Label:", predictions[i].argmax())
 print("Actual Label:", test_labels[i])