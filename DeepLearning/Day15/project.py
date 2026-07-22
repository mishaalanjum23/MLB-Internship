import cv2
import numpy as np
import os

input_folder = "input"
output_folder = "output_images"

os.makedirs(output_folder, exist_ok = True)

# Process all images
for filename in os.listdir(input_folder):
    image_path = os.path.join(input_folder, filename)
    img = cv2.imread(image_path)

    if img is None:
        continue

    rows, cols = img.shape[:2]

 # Perspective Transformation
    if filename == "doc10.jpg":
      pts1 = np.float32([
      [95, 80],    
      [650, 45],   
      [70, 960],  
      [730, 920]  
     ])

      pts2 = np.float32([
        [0, 0],
        [500, 0],
        [0, 800],
        [500, 800]
     ])
      M = cv2.getPerspectiveTransform(pts1, pts2)
      result = cv2.warpPerspective(img, M, (500, 800))

    else:
      result = img

 # Convert to Grayscale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

 # Reduce Noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

 # Enhance Brightness & Contrast
    enhanced = cv2.convertScaleAbs(blur, alpha=1.1, beta=0)

 # Sharpen Image
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharp = cv2.filter2D(enhanced, -1, kernel)

 # Save Image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, sharp)

print("All images processed successfully!")