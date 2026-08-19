# YOLOv8 Instance Segmentation

## Overview

A custom instance segmentation dataset was created and used to compare:

- YOLOv8n-seg trained from scratch
- YOLOv8n-seg fine-tuned from pretrained COCO weights

## Dataset

- **Class:** Car
- **Images:** 50
- **Training:** 40 images
- **Validation:** 10 images
- **Test images:** 6
- **Annotation Tool:** CVAT

**Dataset:** [Google Drive Link](https://drive.google.com/drive/folders/1k-diDOvQMbG-RoA7ok2WXhxWJydSzhke?usp=sharing)

## Training

Both models were trained using the same settings:

- Image size: 320
- Batch size: 4
- Epochs: 30
- Device: CPU

## Results

| Metric | From Scratch | Pretrained |
|---|---:|---:|
| Training Time | 6.52 min | 5.64 min |
| Mask mAP50 | 0.263 | **0.995** |
| Mask mAP50-95 | 0.128 | **0.905** |
| Avg. Inference Time | 100.73 ms/image | **93.01 ms/image** |

## Conclusion

The **pretrained model performed significantly better** than the model trained from scratch. It achieved much higher mask mAP because it had already learned useful visual features from the COCO dataset. With only 40 training images, these pretrained features allowed the model to learn the car segmentation task much more effectively.