from ultralytics import YOLO
import os

# Load trained model
model = YOLO(r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day19\best.pt")

# Input and output paths
input_video = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day19\test_video\video.mp4"
output_folder = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day19\test_results"
    
# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Run detection on the video
results = model.predict(
    source=input_video,
    save=True,
    project=output_folder,
    name="prediction",
    conf=0.25
)

print("Video testing complete!")