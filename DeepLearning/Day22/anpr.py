from ultralytics import YOLO
import cv2
import os
import csv
import easyocr

# load models
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO(r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day22\license_plate_detector.pt")

# EasyOCR
reader = easyocr.Reader(["en"])

# folders
input_folder = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day22\test_images"
detection_folder = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day22\detection_results"
crop_folder = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day22\cropped_plates"
processed_folder = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day22\processed_plates"

os.makedirs(detection_folder, exist_ok=True)
os.makedirs(crop_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)


# CSV FILE
csv_path = "anpr_results.csv"

csv_file = open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Image",
    "Vehicle",
    "Plate_Crop",
    "Plate_Text",
    "OCR_Confidence",
    "Plate_Detection_Confidence"
])

# COCO VEHICLE CLASSES
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

vehicle_classes = [2, 3, 5, 7]

# processing each image
for filename in os.listdir(input_folder):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(
        input_folder,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read: {filename}")
        continue

    print(f"\nProcessing: {filename}")

    output_image = image.copy() #keeping this for final output


    # vehicle detection
    vehicle_results = vehicle_model.predict(
        source=image,
        conf=0.30,
        classes=vehicle_classes,
        verbose=False
    )

    vehicle_number = 0
    plate_number = 0

    for vehicle_result in vehicle_results:

        if vehicle_result.boxes is None:
            continue

        for vehicle_box in vehicle_result.boxes:

            vehicle_number += 1

            # Vehicle coordinates
            vx1, vy1, vx2, vy2 = map(
                int,
                vehicle_box.xyxy[0]
            )

            # Keep coordinates inside image
            vx1 = max(0, vx1)
            vy1 = max(0, vy1)
            vx2 = min(image.shape[1], vx2)
            vy2 = min(image.shape[0], vy2)


            vehicle_confidence = float(
                vehicle_box.conf[0]
            )

            # vehicle bounding box
            cv2.rectangle(
                output_image,
                (vx1, vy1),
                (vx2, vy2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                output_image,
                f"Vehicle {vehicle_number} "
                f"{vehicle_confidence:.2f}",
                (vx1, max(25, vy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
            
            # temporary cropping
            vehicle_crop = image[
                vy1:vy2,
                vx1:vx2
            ]

            if vehicle_crop.size == 0:
                continue


            # License plate detection
            plate_results = plate_model.predict(
                source=vehicle_crop,
                conf=0.50,
                verbose=False
            )

            for plate_result in plate_results:

                if plate_result.boxes is None:
                    continue

                for plate_box in plate_result.boxes:

                    plate_number += 1

                    # Coordinates relative to vehicle crop
                    px1, py1, px2, py2 = map(
                        int,
                        plate_box.xyxy[0]
                    )

                    # Convert to original image coordinates
                    plate_x1 = vx1 + px1
                    plate_y1 = vy1 + py1
                    plate_x2 = vx1 + px2
                    plate_y2 = vy1 + py2

                    # Keep inside image
                    plate_x1 = max(0, plate_x1)
                    plate_y1 = max(0, plate_y1)
                    plate_x2 = min(
                        image.shape[1],
                        plate_x2
                    )
                    plate_y2 = min(
                        image.shape[0],
                        plate_y2
                    )

                    plate_detection_confidence = float(
                        plate_box.conf[0]
                    )


                    # draw plate bounding box on output image
                    cv2.rectangle(
                        output_image,
                        (plate_x1, plate_y1),
                        (plate_x2, plate_y2),
                        (0, 255, 0),
                        2
                    )

                    # Crop plate 
                    plate_crop = image[
                        plate_y1:plate_y2,
                        plate_x1:plate_x2
                    ]


                    if plate_crop.size == 0:
                        continue


                    base_name = os.path.splitext(
                        filename
                    )[0]

                    crop_name = (
                        f"{base_name}"
                        f"_vehicle_{vehicle_number}"
                        f"_plate_{plate_number}.jpg"
                    )

                    crop_path = os.path.join(
                        crop_folder,
                        crop_name
                    )

                    cv2.imwrite(
                        crop_path,
                        plate_crop
                    )


                    # Preprocessing plate
                    resized = cv2.resize(
                        plate_crop,
                        None,
                        fx=3,
                        fy=3,
                        interpolation=cv2.INTER_CUBIC
                    )

                    gray = cv2.cvtColor(
                        resized,
                        cv2.COLOR_BGR2GRAY
                    )

                    clahe = cv2.createCLAHE(
                        clipLimit=2.0,
                        tileGridSize=(8, 8)
                    )

                    enhanced = clahe.apply(gray)

                    filtered = cv2.bilateralFilter(
                        enhanced,
                        5,
                        30,
                        30
                    )

                    blur = cv2.GaussianBlur(
                       filtered,
                       (0, 0),
                        1
                    )

                    weighted = cv2.addWeighted(
                        filtered,
                        1.3,
                        blur,
                       -0.3,
                        0
                  )


                    # save processed plate image
                    processed_name = (
                        f"{base_name}"
                        f"_vehicle_{vehicle_number}"
                        f"_plate_{plate_number}"
                        f"_processed.jpg"
                    )

                    processed_path = os.path.join(
                        processed_folder,
                        processed_name
                    )

                    cv2.imwrite(
                        processed_path,
                        weighted
                    )


                    # Applying OCR
                    ocr_results = reader.readtext(
                        weighted
                    )

                    plate_text = "Unreadable"
                    ocr_confidence = 0.0

                    if ocr_results:

                        # Select the result with highest OCR confidence
                        best_result = max(
                            ocr_results,
                            key=lambda x: x[2]
                        )

                        detected_text = best_result[1]
                        detected_confidence = float(
                            best_result[2]
                        )

                        # Clean text
                        cleaned_text = (
                            detected_text
                            .upper()
                            .replace(" ", "")
                            .replace("-", "")
                        )

                        # Decide if readable
                        if (
                            detected_confidence >= 0.20
                            and len(cleaned_text) >= 3
                        ):
                            plate_text = cleaned_text
                            ocr_confidence = (
                                detected_confidence
                            )


                    # Overlay OCR result
                    text_y = max(
                        25,
                        plate_y1 - 10
                    )

                    cv2.putText(
                        output_image,
                        f"Plate: {plate_text}",
                        (plate_x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )


                    # save csv result
                    csv_writer.writerow([
                        filename,
                        vehicle_number,
                        crop_name,
                        plate_text,
                        f"{ocr_confidence:.2f}",
                        f"{plate_detection_confidence:.2f}"
                    ])


                    # print results
                    print(
                        f"  Vehicle {vehicle_number} | "
                        f"Plate {plate_number}"
                    )

                    print(
                        f"  Plate detection confidence: "
                        f"{plate_detection_confidence:.2f}"
                    )

                    print(
                        f"  OCR: {plate_text} | "
                        f"Confidence: "
                        f"{ocr_confidence:.2f}"
                    )


    # save final detection image
    detection_path = os.path.join(
        detection_folder,
        filename
    )

    cv2.imwrite(
        detection_path,
        output_image
    )

csv_file.close()

print("ANPR processing complete!")