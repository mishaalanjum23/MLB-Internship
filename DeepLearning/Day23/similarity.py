import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity


input_folder = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day23\input"
query_img = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day23\input\dog1.jpg"
Topk = 5
output_folder = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day23\output"

os.makedirs(output_folder, exist_ok=True)

# Load pretrained MobileNetV2
model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)


# Extract image embedding
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


# Create embeddings
image_paths = []

for filename in os.listdir(input_folder):

    path = os.path.join(
        input_folder,
        filename
    )

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        image_paths.append(path)


embeddings = []

for path in image_paths:

    print(
        f"Processing: {os.path.basename(path)}"
    )

    embeddings.append(
        get_embedding(path)
    )


embeddings = np.array(embeddings)


# Query image embedding
query_embedding = get_embedding(
    query_img
)


# Calculate cosine similarity
scores = cosine_similarity(
    [query_embedding],
    embeddings
)[0]


# Sort results
results = []

for path, score in zip(
    image_paths,
    scores
):

    # not including the query image 
    if os.path.abspath(path) != os.path.abspath(query_img):

        results.append(
            (path, float(score))
        )


results.sort(
    key=lambda x: x[1],
    reverse=True
)

top_results = results[:Topk]

# Save results to CSV
csv_path = os.path.join(
    output_folder,
    "similarity_results.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Rank",
        "Query Image",
        "Similar Image",
        "Similarity Score"
    ])

    for i, (path, score) in enumerate(
        top_results,
        start=1
    ):

        writer.writerow([
            i,
            os.path.basename(query_img),
            os.path.basename(path),
            round(score, 4)
        ])

# Create results grid
fig, axes = plt.subplots(
    1,
    Topk,
    figsize=(20, 5)
)

for i, (path, score) in enumerate(
    top_results
):

    img = image.load_img(path)

    axes[i].imshow(img)

    axes[i].set_title(
        f"{os.path.basename(path)}\n"
        f"Similarity: {score:.4f}"
    )

    axes[i].axis("off")


plt.suptitle(
    f"Top {Topk} Similar Images to "
    f"{os.path.basename(query_img)}",
    fontsize=16
)

plt.tight_layout()

# Save results grid
grid_path = os.path.join(
    output_folder,
    "similarity_grid.png"
)

plt.savefig(
    grid_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nResults saved successfully!")