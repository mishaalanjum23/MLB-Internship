# Cats vs Dogs Image Classifier using Transfer Learning

## What is Transfer Learning?

Transfer Learning means using a model that has already been trained on a large dataset. Instead of training a CNN from scratch, I used a pre-trained model and added my own classifier.

## Why did I choose MobileNetV2?

I chose MobileNetV2 because it is fast, lightweight, and already trained on ImageNet. It performs well on image classification tasks and is a good choice for transfer learning.

## Final Validation Accuracy

**Validation Accuracy:** **98.88%**

This exceeded the target accuracy of 93%.

## Key Challenges and Lessons Learned

### Challenges

* The Cats vs Dogs dataset was not loading correctly with TensorFlow Datasets, so I used a monkey patch to fix the issue and successfully load the dataset.
* At first, I did not shuffle the dataset. I learned that shuffling helps create a better training and validation split.

### Lessons Learned

* I learned how transfer learning works and how to use a pre-trained MobileNetV2 model.
* I learned how to freeze the base model and add my own classification layers.
* I understood why preprocessing should match the pre-trained model.
* I also learned the importance of shuffling the training data before training.
