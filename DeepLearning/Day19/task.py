import cv2
import os

input_folder = "input_videos"

output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# Save every 5th frame
frame_interval = 5

videos = os.listdir(input_folder)
total_saved = 0

for video in videos:

    video_path = os.path.join(input_folder, video)
    vid = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_count = 0

    video_name = os.path.splitext(video)[0]

    print(f"\nProcessing {video}...")

    while True:
        ret, frame = vid.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"{video_name}_frame_{saved_count:04d}.jpg"
            filepath = os.path.join(output_folder, filename)

            cv2.imwrite(filepath, frame)
            saved_count += 1
            total_saved += 1

        frame_count += 1

    vid.release()

    print(f"Saved {saved_count} frames.")

print(f" Total frames Saved: {total_saved}")
print("\nAll videos processed successfully.")