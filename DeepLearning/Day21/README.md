# Document Text Extraction Tool

## Overview

A document text extraction pipeline that improves image quality using preprocessing, extracts text using EasyOCR with bounding boxes, saves the results as `.json` and `.txt` files, and compares OCR accuracy on raw and preprocessed images.

## Dataset & Setup

- **Dataset:** 13 document images, including 3 difficult cases with heavy tilt/blur
- **OCR Engine:** EasyOCR
- **Preprocessing:** Grayscale conversion, denoising, CLAHE contrast enhancement, and deskewing
- **Outputs:** Annotated images with bounding boxes, `.txt` text files, and `.json` files containing text, confidence scores, and coordinates


## Results

| Metric | Raw Images | Preprocessed Images |
|---|---:|---:|
| Word Error Rate (WER) | 0.8319 | **0.8275** |
| Character Error Rate (CER) | **0.6170** | 0.6203 |


## Conclusion

The preprocessing made a **small improvement in WER**, but CER increased slightly. Overall, preprocessing did not significantly improve OCR accuracy on this dataset. Difficult cases with heavy blur and poor image quality remained challenging for OCR. 