# Object Detection using YOLO

## What is Object Detection?

Object Detection is a computer vision task that identifies objects in an image and also shows their locations using bounding boxes.

## How is it different from Image Classification?

Image Classification only predicts what is present in an image. Object Detection predicts both what the object is and where it is located.

## What is YOLO?

YOLO (You Only Look Once) is a object detection model. It detects multiple objects in an image, draws bounding boxes around them, and provides confidence scores for each detection.

## Dataset Used

I downloaded and used the **Fruits and Vegetables** dataset from Roboflow universe.
[Dataset link](https://universe.roboflow.com/orkhan-aliyev-8nktf/fruits-and-vegetables-2vf7u/dataset/1)

## Objects Detected

The pre-trained YOLO model successfully detected common fruits such as **apple, banana, and orange**.

## Observations

* The model detected common fruits with different confidence scores depending on the image.
* Bounding boxes were generally accurate for correctly detected objects.
* The model did not detect dragon fruit because it was not part of the pre-trained model's classes.
* In some cases, the model incorrectly detected background objects or misclassified some fruits. For instance, it classified background as dining table in one image and detected an apple as sports ball.
