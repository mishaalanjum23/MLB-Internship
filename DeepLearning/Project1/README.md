# Smart Parking Lot Occupancy Analyzer

## Project Overview

This project uses YOLOv8 and OpenCV to detect vehicles and free parking spaces in parking lot images. It also calculates parking occupancy statistics and displays the results on the output images.

## Dataset Used

* parking lot Computer Vision Model (Roboflow)
* Dataset link: https://universe.roboflow.com/abdullah-hvgvv/parking-lot-j4ojc

## Project Workflow

1. Train the YOLOv8 model on the parking lot dataset.
2. Load the trained model.
3. Run predictions on the test images.
4. Detect cars and free parking spaces.
5. Draw colored bounding boxes and labels.
6. Calculate occupied spaces, vacant spaces, and occupancy percentage.
7. Save the final annotated images.

## Technologies Used

* Python
* YOLOv8 (Ultralytics)
* OpenCV

## Results

* Successfully detected cars and free parking spaces.
* Displayed occupied, vacant, and occupancy percentage on each image.
* Saved the final annotated output images.

## Challenges Faced

* Model training took several hours on a CPU.
* Some partially visible vehicles were not detected consistently.

## Future Improvements

* Support real-time parking occupancy detection using live video.
* Extend the system to work with different parking lot layouts.

## Project Structure

Project1/
│── main.py
│── README.md
│── requirements.txt
│── parking_lot_ds/
│── sample_output/
│── final_annotated_results/ 
|── recording.txt
|── best.pt
