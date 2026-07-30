import cv2

# Read the input video
video = cv2.VideoCapture("input_video.mp4")

# Get video properties
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Width:", width)
print("Height:", height)
print("Total Frames:", total_frames)

# saving video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("processed_video.mp4", fourcc, fps, (width, height))

# Process video
while True:
    ret, frame = video.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    coloured = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    writer.write(coloured)

    cv2.imshow("Original Video", frame)
    cv2.imshow("Processed Video", coloured)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
writer.release()
cv2.destroyAllWindows()