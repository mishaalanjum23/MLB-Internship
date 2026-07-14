# Fashion MNIST Image Classifier using CNN

## Why CNNs are better than ANNs for image data

CNNs are better for images because they learn image features like edges, textures, and shapes automatically. ANNs flatten the image at the start, so they lose the spatial information.

## Purpose of convolution and pooling layers

* **Convolution layer:** Detects important features and patterns from the image.
* **Pooling layer:** Reduces the size of the feature maps, making training faster and helping reduce unnecessary information.

## Model architecture

* Input Layer
* Conv2D (32 filters, 3×3, ReLU)
* MaxPooling2D (2×2)
* Flatten Layer
* Dense Layer (128 neurons, ReLU)
* Output Layer (10 neurons, Softmax)

## Model performance

* Training Accuracy: **(0.93)**
* Validation Accuracy: **(0.91)**
* Test Accuracy: **(0.90)**

I also included the training/validation accuracy graph and the confusion matrix.

## Challenges faced and how I solved them

I got an input shape error because the CNN expected images with a channel dimension. I fixed it by reshaping the images from (28, 28) to (28, 28, 1) before training the model.
