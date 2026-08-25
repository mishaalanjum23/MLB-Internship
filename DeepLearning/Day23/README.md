# Similar & Duplicate Image Finder

## Overview

Built a Similar & Duplicate Image Finder that finds visually similar images using a pretrained CNN and detects exact/near-duplicate images using perceptual hashing.

## Dataset

The dataset contains 24 images:

- 5 dog images
- 5 library/books images
- 5 flower images
- 5 road images
- 4 cycle images used for the near-duplicate challenge

The dataset contains different, visually similar, and near-duplicate images.

## Methods Used

- **MobileNetV2** – for extracting image feature embeddings.
- **Cosine Similarity** – for finding the top 5 most similar images for a query image.
- **pHash (Perceptual Hashing)** – for detecting possible exact/near-duplicate images.
- **MobileNetV2 Similarity** – used to verify duplicate candidates.

## Mandatory Challenge

Three modified versions of the cycle image were created:

- Resized
- Cropped
- Brightness-changed

All three modified versions were successfully detected as near-duplicates of the original cycle image.

## Results

The system:

- Finds the top 5 most similar images for a given query image.
- Calculates and saves similarity scores.
- Detects exact/near-duplicate images.
- Verifies duplicate candidates using CNN similarity.
- Saves similarity and duplicate result grids and CSV files.
