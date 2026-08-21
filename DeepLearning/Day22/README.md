# Automatic Number Plate Recognition System (ANPR)

## Overview

Built an Automatic Number Plate Recognition (ANPR) system that detects vehicles, detects license plates, preprocesses plate images, and extracts plate text using OCR.

## Models Used

- **YOLOv8n (COCO)** – for vehicle detection.
- **Pretrained License Plate YOLO model** – for license plate detection.
- **EasyOCR** – for license plate text recognition.

## Preprocessing

The detected license plates were:
- Resized
- Converted to grayscale
- Enhanced using CLAHE for better contrast
- Filtered using bilateral filtering to reduce noise
- Sharpened to improve character visibility

## Results

The system:
- Detects vehicles and license plates.
- Crops and preprocesses detected plates.
- Recognizes plate text using EasyOCR.
- Marks plates that cannot be reliably recognized as **"Unreadable"**.
- Overlays the recognized text on the output image.
- Saves plate information and confidence scores in a CSV file.

## Difficult Case Testing

Three difficult plates were tested, including blurry and angled plates.

- Some plates were completely **Unreadable** to OCR.
- One plate was **partially recognized but contained incorrect characters**.
- Some angled plates were readable to the human eye but could not be reliably recognized by OCR.

## Possible Improvements

- Use higher-resolution images for blurry or distant plates.
- Apply advanced deblurring techniques.
- Use perspective correction for angled plates.
- Use a specialized license plate OCR model.