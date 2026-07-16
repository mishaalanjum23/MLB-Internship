from ultralytics import YOLO
model = YOLO("yolov8n.pt")

results = model(r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day13\FruitsandVegetables\valid\images", save = True)

for result in results:
    for box in result.boxes:
        class_name = model.names[int(box.cls.item())]
        print("Object:", class_name)
        print("Confidence:", box.conf.item())
        print("Bounding Box:", box.xyxy)