import os
os.environ["FLAGS_enable_pir_api"] = "0"

from ultralytics import YOLO
import cv2
import csv
from paddleocr import PaddleOCR


# load models
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO(r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day22\license_plate_detector.pt")

# PaddleOCR
ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False
)

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

# COCO VEHICLE CLASSES(2 = car, 3 = motorcycle, 5 = bus, 7 = truck)

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

    output_image = image.copy()

    # Vehicle detection
    vehicle_results = vehicle_model.predict(
        source=image,
        conf=0.30,
        classes=vehicle_classes,
        verbose=False
    )

    vehicle_number = 0
    plate_number = 0

    # plate detections from all vehicles
    all_plate_detections = []

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

            # Temporary vehicle cropping
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

                if len(plate_result.boxes) == 0:
                    continue

                for plate_box in plate_result.boxes:

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

                    # Store the plate detection
                    all_plate_detections.append({
                        "vehicle_number": vehicle_number,
                        "x1": plate_x1,
                        "y1": plate_y1,
                        "x2": plate_x2,
                        "y2": plate_y2,
                        "confidence": plate_detection_confidence
                    })

    # Global NMS applied across all vehicles
    if len(all_plate_detections) == 0:
        continue

    nms_boxes = []
    nms_scores = []

    for detection in all_plate_detections:

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        width = x2 - x1
        height = y2 - y1

        nms_boxes.append([
            x1,
            y1,
            width,
            height
        ])

        nms_scores.append(
            detection["confidence"]
        )

    keep_indices = cv2.dnn.NMSBoxes(
        nms_boxes,
        nms_scores,
        score_threshold=0.50,
        nms_threshold=0.40
    )

    if len(keep_indices) == 0:
        continue

    # Process remaining plate detections
    for index in keep_indices:

        index = int(index)

        detection = all_plate_detections[index]

        vehicle_number = detection["vehicle_number"]

        plate_x1 = detection["x1"]
        plate_y1 = detection["y1"]
        plate_x2 = detection["x2"]
        plate_y2 = detection["y2"]

        plate_detection_confidence = (
            detection["confidence"]
        )

        plate_number += 1

        # Draw ONLY plate bounding box
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

        enhanced = clahe.apply(
            gray
        )

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

        # Save processed plate image
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

        # PaddleOCR
        # Trying more than one version of the plate.

        ocr_variants = [

            # Original resized grayscale
            gray,

            # CLAHE enhanced
            enhanced,

            # Final sharpened image
            weighted

        ]

        best_text = None
        best_confidence = 0.0

        for variant in ocr_variants:

            # PaddleOCR expects a 3-channel image
            ocr_image = cv2.cvtColor(
                variant,
                cv2.COLOR_GRAY2BGR
            )

            ocr_results = ocr.predict(
                ocr_image
            )

            for result in ocr_results:

                result_data = result

                try:

                    texts = result_data[
                        "rec_texts"
                    ]

                    scores = result_data[
                        "rec_scores"
                    ]

                except:

                    try:

                        texts = result_data[
                            "res"
                        ][
                            "rec_texts"
                        ]

                        scores = result_data[
                            "res"
                        ][
                            "rec_scores"
                        ]

                    except:

                        texts = []
                        scores = []

                for text, confidence in zip(
                    texts,
                    scores
                ):

                    confidence = float(
                        confidence
                    )

                    if confidence > best_confidence:

                        best_text = text

                        best_confidence = (
                            confidence
                        )

        # OCR result
        plate_text = "Unreadable"
        ocr_confidence = 0.0

        if best_text is not None:

            detected_text = best_text

            detected_confidence = (
                best_confidence
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
                detected_confidence >= 0.60
                and len(cleaned_text) >= 3
            ):

                plate_text = cleaned_text

                ocr_confidence = (
                    detected_confidence
                )

        # Overlay ONLY plate OCR result
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

        # Save CSV result
        csv_writer.writerow([
            filename,
            vehicle_number,
            crop_name,
            plate_text,
            f"{ocr_confidence:.2f}",
            f"{plate_detection_confidence:.2f}"
        ])

        # Print results
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

    # Save final detection image
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