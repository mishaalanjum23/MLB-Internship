from ultralytics import YOLO
import time

model = YOLO("yolov8n-seg.pt")

start_time = time.time()

model.train(
    data="C:\\Users\\sunny\\OneDrive\\Desktop\\MLB-Internship\\DeepLearning\\Day20\\car_segmentation_dataset\\data.yaml",
    epochs=30,
    imgsz=320,
    batch=4,
    project="results",
    name="car_segmentation_pretrained"
)

training_time = time.time() - start_time

print(f"Training time: {training_time / 60:.2f} minutes")