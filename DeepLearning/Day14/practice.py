import cv2
import numpy as np
import os

# Reading image and displaying info
image = cv2.imread(r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day14\dog.jpg")

height, width, channels = image.shape
size = os.path.getsize(r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day14\dog.jpg")
 
print("Height:", height)
print("Width:", width)
print("Channels:", channels)
print("File Size:", size)

# 2. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/grayscale.jpg", gray)

# 3. Resize image
small = cv2.resize(image, (300, 300))
large = cv2.resize(image, (700, 700))

cv2.imwrite("output/small.jpg", small)
cv2.imwrite("output/large.jpg", large)

# 4. Crop image
crop1 = image[0:300, 0:300]
crop2 = image[200:500, 200:500]

cv2.imwrite("output/crop1.jpg", crop1)
cv2.imwrite("output/crop2.jpg", crop2)

# 5. Rotate image
rotate90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
rotate180 = cv2.rotate(image, cv2.ROTATE_180)
rotate270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imwrite("output/rotate90.jpg", rotate90)
cv2.imwrite("output/rotate180.jpg", rotate180)
cv2.imwrite("output/rotate270.jpg", rotate270)

# 6. Flip image
horizontal = cv2.flip(image, 1)
vertical = cv2.flip(image, 0)

cv2.imwrite("output/horizontal.jpg", horizontal)
cv2.imwrite("output/vertical.jpg", vertical)

# 7. Draw shapes and add text
draw = image.copy()

# Rectangle
cv2.rectangle(draw, (30, 30), (200, 150), (0, 255, 0), 2)

# Circle
cv2.circle(draw, (350, 150), 50, (255, 0, 0), 2)

# Line
cv2.line(draw, (0, 0), (400, 300), (0, 0, 255), 2)

# Polygon
points = np.array([[450, 50], [550, 100], [500, 200], [400, 180]], np.int32)
cv2.polylines(draw, [points], True, (255, 255, 0), 2)

# writing Text on img
cv2.putText(draw, "Hello", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.putText(draw, "21 July 2026", (20, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

cv2.imwrite("output/final_image.jpg", draw)
print("Images saved successfully.")