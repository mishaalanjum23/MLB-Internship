# Automatic Number Plate Recognition System (ANPR)

## Overview

Built an Automatic Number Plate Recognition (ANPR) system that detects vehicles, detects license plates, preprocesses plate images, and extracts plate text using OCR.

## Models Used

- **YOLOv8n (COCO)** – for vehicle detection.
- **Pretrained License Plate YOLO model** – for license plate detection.
- **PaddleOCR** – for license plate text recognition.

## Preprocessing

The detected license plates were:
- Resized
- Converted to grayscale
- Enhanced using CLAHE for better contrast
- Filtered using bilateral filtering to reduce noise
- Sharpened using weighted image enhancement

Multiple processed versions of the plate images were passed to PaddleOCR, and the result with the highest confidence was selected.

## Results

The system:
- Detects vehicles and license plates.
- Crops and preprocesses detected plates.
- Recognizes plate text using PaddleOCR.
- Marks plates that cannot be reliably recognized as **"Unreadable"**.
- Overlays the recognized text on the output image.
- Saves cropped and processed plate images.
- Saves plate information and confidence scores in a CSV file.

## Difficult Case Testing

Three difficult plates were tested, including blurry and angled plates.

- Some plates were initially difficult for OCR to recognize.
- Testing multiple preprocessing versions improved OCR results.
- Angled and lower-quality plates could still be recognized.

## Possible Improvements

- Use higher-resolution images for blurry or distant plates.
- Apply advanced deblurring techniques.
- Use perspective correction for severely angled plates.
- Use a specialized license plate OCR model.