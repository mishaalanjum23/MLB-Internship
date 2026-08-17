# Traffic Object Detection, Tracking and Counting

## Overview

This project focuses on object detection, tracking, and counting in traffic videos using a custom-trained YOLOv8-n model.

## Work Completed

* Collected raw traffic videos with different camera angles and lighting conditions.
* Extracted frames from the videos.
* Manually annotated the frames using a custom annotation tool.
* Prepared the annotated data in YOLO format.
* Trained a YOLOv8-n model from scratch for traffic object detection.
* Tested the trained model on a new, unseen traffic video.
* Implemented object tracking using ByteTrack with the trained YOLO model.
* Added unique tracking IDs and visual movement trails for tracked objects.
* Added line-based counting to count objects crossing the defined line by class.

## Classes

The model was trained to detect:

* Car
* Person
* Bike
* Truck
* Bus

## Results

**Object Detection Video:**
https://drive.google.com/file/d/1065iyM9NgkZjamy5E0xG8yiVdBj6I0-d/view?usp=sharing

**Object Tracking Video:**
https://drive.google.com/file/d/1Xtf0_PZRCalxCREJ-wC6DnpiLCrrgiNs/view?usp=sharing