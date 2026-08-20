import cv2
import numpy as np
import os

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def preprocess_image(image):
    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Denoising
    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=3,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 3. Contrast correction
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(denoised)

    # 4. Deskew
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    deskewed = deskew(enhanced_bgr)

    return deskewed

input_folder = "input_images/images"
output_folder = "preprocessed_images"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process every image in the input folder
for filename in os.listdir(input_folder):

    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        image = cv2.imread(input_path)

        if image is None:
            print(f"Could not read: {filename}")
            continue

        processed_image = preprocess_image(image)

        cv2.imwrite(output_path, processed_image)

        print(f"Processed: {filename}")

print("\nAll images processed successfully!")
