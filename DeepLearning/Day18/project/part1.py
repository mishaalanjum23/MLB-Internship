import cv2
import os

input_folder = "input_videos"
output_folder = "output_videos"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"processed_{filename}")

    video = cv2.VideoCapture(input_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing: {filename}")
    print("FPS:", fps)
    print("Width:", width)
    print("Height:", height)
    print("Total Frames:", total_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = video.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 100, 200)

        coloured = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        writer.write(coloured)

        cv2.imshow("Original Video", frame)
        cv2.imshow("Processed Video", edges)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    writer.release()

cv2.destroyAllWindows()

print("\nAll videos processed successfully!")
