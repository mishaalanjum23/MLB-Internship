from collections import defaultdict
import os
import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

video_path = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day19\test_video\video.mp4"

output_folder = "test_results"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "tracking_output.mp4")

# Open video
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Tracking history
track_history = defaultdict(list)
MAX_TAIL_LENGTH = 12

# Standardize counts dictionary keys in lowercase
counts = defaultdict(int)

# Set of already counted IDs
counted_ids = set()

# Map of track_id -> side string ("above" or "below")
object_side = {}

# Line position (70% of frame height)
line_y = int(height * 0.7)

# Process video
while True:
  ret, frame = cap.read()
  if not ret:
    break

  # YOLO + ByteTrack
  results = model.track(
      frame, persist=True, tracker="bytetrack.yaml", verbose=False
  )
  result = results[0]

  # Draw reference line
  cv2.line(frame, (0, line_y), (width, line_y), (255, 255, 255), 2)

  if result.boxes.id is not None:
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    track_ids = result.boxes.id.cpu().numpy().astype(int)

    for box, class_id, track_id in zip(boxes, class_ids, track_ids):
      x1, y1, x2, y2 = map(int, box)

      # Center point of bounding box
      center_x = (x1 + x2) // 2
      center_y = (y1 + y2) // 2

      # Normalize class name to lowercase to prevent key mismatches
      class_name = str(model.names[class_id]).lower().strip()

      # Tracking tail visualization
      track_history[track_id].append((center_x, center_y))
      if len(track_history[track_id]) > MAX_TAIL_LENGTH:
        track_history[track_id].pop(0)

      points = track_history[track_id]
      for i in range(1, len(points)):
        cv2.line(
            frame, points[i - 1], points[i], (255, 180, 80), 2
        )

      # Draw Bounding Box & Label
      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(
          frame,
          f"{class_name} ID:{track_id}",
          (x1, max(y1 - 10, 20)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 255, 0),
          2,
      )

      # Determine current side relative to line
      current_side = "above" if center_y < line_y else "below"

      # Record side on first occurrence
      if track_id not in object_side:
        object_side[track_id] = current_side

      # Check for side transition (Supports BOTH top->bottom & bottom->top crossing)
      elif (
          object_side[track_id] != current_side and track_id not in counted_ids
      ):
        counted_ids.add(track_id)
        counts[class_name] += 1
        # Update current side to reflect state shift
        object_side[track_id] = current_side

  # Render counts on output video
  y_offset = 30
  for name, count in counts.items():
    cv2.putText(
        frame,
        f"{name}: {count}",
        (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
    )
    y_offset += 30

  out.write(frame)

cap.release()
out.release()

print("\nFinal counts:")
for name, count in counts.items():
  print(f"{name}: {count}")

print("Tracking complete")