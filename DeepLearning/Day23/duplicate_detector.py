import os
import csv
import imagehash
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

input_folder = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day23\input"
query_img = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day23\input\cycle.jpg"
output_folder = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day23\output"

PHASH_THRESHOLD = 22       # pHash threshold for finding possible duplicates
CNN_THRESHOLD = 0.55       # CNN similarity threshold for verification

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Load MobileNetV2
model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# Calculate pHash
def calculate_phash(image_path):

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return imagehash.phash(img)


# Calculate CNN embedding
def get_embedding(image_path):

    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

    embedding = model.predict(
        img_array,
        verbose=0
    )

    return embedding[0]


# Query hashes and embedding
query_hash = calculate_phash(query_img)
query_embedding = get_embedding(query_img)


# Find pHash candidates
candidates = []

for filename in os.listdir(input_folder):

    path = os.path.join(
        input_folder,
        filename
    )

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    if os.path.abspath(path) == os.path.abspath(query_img):
        continue

    current_hash = calculate_phash(path)

    hash_distance = query_hash - current_hash

    if hash_distance <= PHASH_THRESHOLD:

        candidates.append(
            (path, hash_distance)
        )

# Verify candidates using CNN
duplicates = []

for path, hash_distance in candidates:

    current_embedding = get_embedding(path)

    cnn_score = cosine_similarity(
        [query_embedding],
        [current_embedding]
    )[0][0]

    if cnn_score >= CNN_THRESHOLD:

        if hash_distance == 0:
            result = "Exact Duplicate"
        else:
            result = "Near-Duplicate"

        duplicates.append(
            (
                path,
                hash_distance,
                float(cnn_score),
                result
            )
        )

# Sort by CNN similarity
duplicates.sort(
    key=lambda x: x[2],
    reverse=True
)

# Display results
print("\nDuplicate / Near-Duplicate Results:\n")

if duplicates:

    for path, hash_distance, cnn_score, result in duplicates:

        print(
            f"{os.path.basename(path)} "
            f"-> pHash Distance: {hash_distance} "
            f"-> CNN Similarity: {cnn_score:.4f} "
            f"-> {result}"
        )

else:

    print("No verified duplicates found.")

# Save CSV
csv_path = os.path.join(
    output_folder,
    "duplicate_results.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Query Image",
        "Matching Image",
        "pHash Distance",
        "CNN Similarity",
        "Result"
    ])

    for path, hash_distance, cnn_score, result in duplicates:

        writer.writerow([
            os.path.basename(query_img),
            os.path.basename(path),
            hash_distance,
            round(cnn_score, 4),
            result
        ])

# Create results grid
if duplicates:

    grid_items = [
        (
            query_img,
            0,
            1.0,
            "Original / Query"
        )
    ]

    grid_items.extend(duplicates)

    number_of_images = len(grid_items)

    fig, axes = plt.subplots(
        1,
        number_of_images,
        figsize=(5 * number_of_images, 5)
    )

    if number_of_images == 1:
        axes = [axes]

    for i, item in enumerate(grid_items):

        path = item[0]
        hash_distance = item[1]
        cnn_score = item[2]

        if i == 0:

            title = (
                f"{os.path.basename(path)}\n"
                f"Original / Query"
            )

        else:

            result = item[3]

            title = (
                f"{os.path.basename(path)}\n"
                f"pHash: {hash_distance}\n"
                f"CNN: {cnn_score:.4f}\n"
                f"{result}"
            )

        img = Image.open(path)

        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.suptitle(
        "Duplicate / Near-Duplicate Detection",
        fontsize=16
    )

    plt.tight_layout()

    grid_path = os.path.join(
        output_folder,
        "duplicate_grid.png"
    )

    plt.savefig(
        grid_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

else:

    grid_path = None

print("\nResults saved successfully!") 