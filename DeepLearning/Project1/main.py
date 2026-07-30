from ultralytics import YOLO
import cv2
import os

dataset = "C:\\Users\\sunny\\OneDrive\\Desktop\\MLB-Internship\\DeepLearning\\Project1\\parking_lot_ds\\data.yaml"

# Training
# model = YOLO("yolov8n.pt")
# model.train(data = dataset, epochs = 15, imgsz = 640, batch = 8)

model = YOLO(r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Project1\train\weights\best.pt")

# Validation
#evaluate = model.val(data = dataset)
# print("mAP50:", evaluate.box.map50)
# print("mAP50-95:", evaluate.box.map)

# predictions
results = model.predict(source = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Project1\parking_lot_ds\test\images", save=True, conf=0.5, show_conf=False)

output_folder = "final_annotated_results"
os.makedirs(output_folder, exist_ok=True)

for result in results:

    image = result.orig_img.copy()

    car_count = 0
    free_count = 0

    boxes = result.boxes.xyxy
    classes = result.boxes.cls

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)

        if int(cls) == 0: 
            label = "Car"
            color = (0, 0, 255)
            car_count += 1

        else:
            label = "Free"
            color = (0, 255, 0)   
            free_count += 1

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    total_slots = car_count + free_count
    if total_slots > 0:
        occupancy = (car_count / total_slots) * 100
    else:
        occupancy = 0

    cv2.putText(image, f"Occupied : {car_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(image, f"Vacant : {free_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"Occupancy : {occupancy:.1f}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    image_name = os.path.basename(result.path)
    cv2.imwrite(os.path.join(output_folder, image_name),image)