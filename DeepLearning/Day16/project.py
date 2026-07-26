import cv2
import numpy as np
import os

input_folder = "input_images"
output_folder = "output_images"

os.makedirs(output_folder, exist_ok = True)

for filename in os.listdir(input_folder):

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        continue

    # Remove extension from filename
    name = os.path.splitext(filename)[0]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Detect edges
    edges = cv2.Canny(blur, 50, 150)

    # Morphological operation(Closing)
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw largest contour
    final_image = image.copy()

    if contours:
        largest_contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(final_image, [largest_contour], -1, (0, 255, 0), 3)

    # Save all outputs
    cv2.imwrite(os.path.join(output_folder, f"{name}_original.jpg"), image)
    cv2.imwrite(os.path.join(output_folder, f"{name}_edges.jpg"), edges)
    cv2.imwrite(os.path.join(output_folder, f"{name}_morph.jpg"), morph)
    cv2.imwrite(os.path.join(output_folder, f"{name}_final.jpg"), final_image)

print("Done!")
