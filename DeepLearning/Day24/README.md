# Caption & Search Photo Gallery

## Overview

A photo gallery tool that automatically generates captions for images using **BLIP** and allows searching images using natural-language queries with **CLIP**.

The tool returns the **top 5 most similar images** for a given text query.

## Features

- Generates image captions using pre-trained **BLIP**
- Creates image embeddings using **CLIP**
- Searches images using natural-language queries
- Returns the top 5 matching images
- Displays each result with:
  - Image
  - Similarity score
  - BLIP caption
- Displays results in an image grid
- Saves search results in **CSV and JSON**
- Saves CLIP image embeddings for reuse
- Supports multiple queries in one run

## Models Used

- **BLIP:** `Salesforce/blip-image-captioning-base`
- **CLIP:** `openai/clip-vit-base-patch32`

Base versions were used for faster processing on CPU.

## Dataset

- 21 images containing different objects, people, activities, food, vehicles, and scenes.
- Multiple examples of some categories were included to make similarity search more meaningful.

## Abstract Query Results

| Query | Result |
|---|---|
| Something to eat | Mostly matched |
| A beautiful scene | Matched |
| An outdoor activity | Partially matched |
