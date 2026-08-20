import easyocr
import cv2
import os
import json
import numpy as np

# OCR READER
reader = easyocr.Reader(['en'], gpu=False)

# FOLDERS
raw_folder = "input_images/images"
preprocessed_folder = "preprocessed_images"

raw_output_folder = "raw_ocr_output"
preprocessed_output_folder = "preprocessed_ocr_output"

os.makedirs(raw_output_folder, exist_ok=True)
os.makedirs(preprocessed_output_folder, exist_ok=True)

# OCR FUNCTION
def perform_ocr(image_path, output_folder):

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read: {image_path}")
        return

    # Run EasyOCR
    results = reader.readtext(image)

    ocr_data = []
    extracted_text = []

    for detection in results:

        box, text, confidence = detection

        # Convert coordinates to integers
        points = [
            [int(point[0]), int(point[1])]
            for point in box
        ]

        # Save OCR information
        ocr_data.append({
            "text": text,
            "coordinates": points,
            "confidence": float(confidence)
        })

        extracted_text.append(text)

        # Draw bounding box
        points_np = np.array(points, dtype=np.int32)

        cv2.polylines(
            image,
            [points_np],
            True,
            (0, 255, 0),
            2
        )

        # Draw confidence score
        x = points[0][0]
        y = points[0][1]

        label = f"{confidence * 100:.1f}%"

        cv2.putText(
            image,
            label,
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # FILENAMES
    filename = os.path.basename(image_path)
    name, extension = os.path.splitext(filename)

    # SAVE ANNOTATED IMAGE

    annotated_path = os.path.join(
        output_folder,
        f"{name}_ocr{extension}"
    )

    cv2.imwrite(annotated_path, image)

    # SAVE JSON
    json_path = os.path.join(
        output_folder,
        f"{name}_ocr.json"
    )

    with open(json_path, "w", encoding="utf-8") as file:

        json.dump(
            ocr_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    # SAVE TEXT
    text_path = os.path.join(
        output_folder,
        f"{name}_ocr.txt"
    )

    with open(text_path, "w", encoding="utf-8") as file:

        file.write(" ".join(extracted_text))


    print(f"OCR completed: {filename}")

# PROCESS RAW IMAGES
print("\nProcessing RAW images...\n")

for filename in os.listdir(raw_folder):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    ):

        image_path = os.path.join(
            raw_folder,
            filename
        )

        perform_ocr(
            image_path,
            raw_output_folder
        )

# PROCESS PREPROCESSED IMAGES
print("\nProcessing PREPROCESSED images...\n")

for filename in os.listdir(preprocessed_folder):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    ):

        image_path = os.path.join(
            preprocessed_folder,
            filename
        )

        perform_ocr(
            image_path,
            preprocessed_output_folder
        )

print("\nOCR processing completed!") 