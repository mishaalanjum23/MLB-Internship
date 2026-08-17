import os
import shutil

frames_folder = "frames"
labels_folder = "labels"
dataset_folder = "dataset"

# 80% videos for training, 20% videos for validation
train_videos = range(1, 13)   # Video 1 to Video 12
val_videos = range(13, 15)    # Video 13 and 14

# Create dataset folders
os.makedirs(os.path.join(dataset_folder, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(dataset_folder, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(dataset_folder, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(dataset_folder, "labels", "val"), exist_ok=True)

train_count = 0
val_count = 0
missing_labels = 0

# Go through all frames
for filename in os.listdir(frames_folder):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    parts = filename.split("_")

    video_part = parts[0] 

    # Get video number
    video_number = int(video_part.replace("video", ""))

    # Corresponding label filename
    image_name = os.path.splitext(filename)[0]
    label_filename = image_name + ".txt"
    label_path = os.path.join(labels_folder, label_filename)

    # Check that the label exists
    if not os.path.exists(label_path):
        print("Missing label:", filename)
        missing_labels += 1
        continue

    # Decide train or validation
    if video_number in train_videos:
        image_destination = os.path.join(
            dataset_folder, "images", "train", filename
        )

        label_destination = os.path.join(
            dataset_folder, "labels", "train", label_filename
        )

        train_count += 1

    elif video_number in val_videos:
        image_destination = os.path.join(
            dataset_folder, "images", "val", filename
        )

        label_destination = os.path.join(
            dataset_folder, "labels", "val", label_filename
        )

        val_count += 1

    else:
        continue

    # Copy image and label
    shutil.copy2(
        os.path.join(frames_folder, filename),
        image_destination
    )

    shutil.copy2(
        label_path,
        label_destination
    )


# Create data.yaml
yaml_content = """path: dataset

train: images/train
val: images/val

names:
  0: Car
  1: Person
  2: Bike
  3: Truck
  4: Bus
"""

with open(
    os.path.join(dataset_folder, "data.yaml"),
    "w",
    encoding="utf-8"
) as file:
    file.write(yaml_content)


print("\nDataset preparation complete!")
print("Training images:", train_count)
print("Validation images:", val_count)
print("Missing labels:", missing_labels)