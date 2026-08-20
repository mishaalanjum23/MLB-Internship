import os
import json
from jiwer import wer, cer


# Folders
annotation_folder = "input_images/annotations"
raw_ocr_folder = "raw_ocr_output"
processed_ocr_folder = "preprocessed_ocr_output"


def get_ground_truth(annotation_path):

    with open(annotation_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    text = []

    for item in data["form"]:
        for word in item["words"]:

            word_text = word["text"].strip()

            if word_text:
                text.append(word_text)

    return " ".join(text)


def read_text(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


# Store results
raw_wer = []
processed_wer = []

raw_cer = []
processed_cer = []


# Process annotations
for filename in os.listdir(annotation_folder):

    if not filename.endswith(".json"):
        continue

    name = os.path.splitext(filename)[0]

    annotation_path = os.path.join(
        annotation_folder,
        filename
    )

    raw_path = os.path.join(
        raw_ocr_folder,
        f"{name}_ocr.txt"
    )

    processed_path = os.path.join(
        processed_ocr_folder,
        f"{name}_ocr.txt"
    )

    # Skip if OCR output doesn't exist
    if not os.path.exists(raw_path):
        continue

    if not os.path.exists(processed_path):
        continue


    # Get texts
    ground_truth = get_ground_truth(annotation_path)

    raw_text = read_text(raw_path)

    processed_text = read_text(processed_path)


    # Calculate WER
    raw_wer.append(
        wer(ground_truth, raw_text)
    )

    processed_wer.append(
        wer(ground_truth, processed_text)
    )


    # Calculate CER
    raw_cer.append(
        cer(ground_truth, raw_text)
    )

    processed_cer.append(
        cer(ground_truth, processed_text)
    )


# Calculate averages
average_raw_wer = sum(raw_wer) / len(raw_wer)
average_processed_wer = sum(processed_wer) / len(processed_wer)

average_raw_cer = sum(raw_cer) / len(raw_cer)
average_processed_cer = sum(processed_cer) / len(processed_cer)

# Print results
print("OCR ACCURACY COMPARISON")

print(f"Images evaluated: {len(raw_wer)}")

print("\nWER:")
print(f"Raw:          {average_raw_wer:.4f}")
print(f"Preprocessed: {average_processed_wer:.4f}")

print("\nCER:")
print(f"Raw:          {average_raw_cer:.4f}")
print(f"Preprocessed: {average_processed_cer:.4f}")
