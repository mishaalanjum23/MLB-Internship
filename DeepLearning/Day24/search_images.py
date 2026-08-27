import os
import re
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image
from transformers import CLIPProcessor, CLIPModel


IMAGE_FOLDER = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day24\input"
RESULTS_FOLDER = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day24\results"

CAPTION_FILE = os.path.join(
    RESULTS_FOLDER,
    "image_captions.csv"
)

EMBEDDING_FILE = os.path.join(
    RESULTS_FOLDER,
    "image_embeddings.npy"
)

SEARCH_CSV = os.path.join(
    RESULTS_FOLDER,
    "search_results.csv"
)

SEARCH_JSON = os.path.join(
    RESULTS_FOLDER,
    "search_results.json"
)

os.makedirs(RESULTS_FOLDER, exist_ok=True)


# Load captions
captions_df = pd.read_csv(CAPTION_FILE)


# Load CLIP
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.eval()


# Compute / Load image embeddings
if os.path.exists(EMBEDDING_FILE):

    print("Loading saved image embeddings...")

    image_embeddings = np.load(
        EMBEDDING_FILE
    )

    # Check that the number of embeddings matches the number of images/captions.
    if len(image_embeddings) != len(captions_df):

        print(
            "\nError: Number of saved embeddings does not "
            "match the number of image captions."
        )

        print(
            "Delete image_embeddings.npy and run the program again."
        )

        raise SystemExit

else:

    print("Computing image embeddings...")

    embeddings = []

    for _, row in captions_df.iterrows():

        filename = row["image"]

        image_path = os.path.join(
            IMAGE_FOLDER,
            filename
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            with torch.no_grad():

                image_features = model.get_image_features(
                    **inputs
                )

                # For newer Transformers versions, get_image_features() returns an object.
                image_features = image_features.pooler_output

            # Normalize embedding
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True
            )

            embeddings.append(
                image_features.squeeze(0).cpu().numpy()
            )

            print(
                f"Processed: {filename}"
            )

        except Exception as e:

            print(
                f"Error processing {filename}: {e}"
            )

    # Make sure every image was processed
    if len(embeddings) != len(captions_df):

        print(
            f"\nWARNING: {len(captions_df) - len(embeddings)} "
            f"image(s) could not be processed."
        )

        print(
            "Embeddings were NOT saved. "
            "Fix the errors and run again."
        )

        raise SystemExit

    image_embeddings = np.array(
        embeddings
    )

    np.save(
        EMBEDDING_FILE,
        image_embeddings
    )

    print(
        f"\nEmbeddings saved to: {EMBEDDING_FILE}"
    )


# Search function
def search_images(query, top_k=5):

    print(
        f'\nSearching for: "{query}"'
    )

    # Convert text query into CLIP embedding
    text_inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        text_features = model.get_text_features(
            **text_inputs
        )

        # Get actual text embedding
        text_features = text_features.pooler_output

    # Normalize text embedding
    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True
    )

    text_embedding = (
        text_features
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # Calculate cosine similarity
    similarities = np.dot(
        image_embeddings,
        text_embedding
    )

    # Get top 5 results
    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        filename = captions_df.iloc[index]["image"]

        caption = captions_df.iloc[index]["caption"]

        score = float(
            similarities[index]
        )

        results.append({
            "rank": rank,
            "image": filename,
            "similarity": round(score, 4),
            "caption": caption
        })

    return results


# Create safe filename from query
def make_safe_filename(query):

    # Convert spaces to underscores
    filename = query.strip().replace(
        " ",
        "_"
    )

    # Remove characters that are not suitable for Windows filenames
    filename = re.sub(
        r'[<>:"/\\|?*]',
        "",
        filename
    )

    # Keep filename reasonably short
    filename = filename[:100]
    return f"search_{filename}.png"


# Display and save result grid
def display_and_save_grid(
    query,
    results
):

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(20, 5)
    )

    # Make axes iterable if there is only one result
    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(
        axes,
        results
    ):

        image_path = os.path.join(
            IMAGE_FOLDER,
            result["image"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        ax.imshow(image)

        ax.axis("off")

        ax.set_title(
            f"{result['image']}\n"
            f"Score: {result['similarity']}\n"
            f"{result['caption']}",
            fontsize=9
        )

    plt.suptitle(
        f'Search Results: "{query}"',
        fontsize=14
    )

    plt.tight_layout()

    # Save grid automatically
    grid_filename = make_safe_filename(
        query
    )

    grid_path = os.path.join(
        RESULTS_FOLDER,
        grid_filename
    )

    plt.savefig(
        grid_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Grid saved to: {grid_path}"
    )

    plt.show()
    plt.close()


# Multiple search queries
all_results = []

while True:

    query = input(
        '\nEnter your search query '
        '(or type "exit" to finish): '
    ).strip()

    # Stop searching
    if query.lower() == "exit":

        break

    # Ignore empty queries
    if not query:

        print(
            "Please enter a query."
        )

        continue

    # Search
    results = search_images(
        query
    )

    # Display results in terminal
    print(
        "\nTop 5 Results:"
    )

    print(
        "=" * 60
    )

    for result in results:

        print(
            f"\n{result['rank']}. "
            f"{result['image']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']}"
        )

        print(
            f"Caption: "
            f"{result['caption']}"
        )

    # Save/display grid
    display_and_save_grid(
        query,
        results
    )

    # Add query to overall results
    for result in results:

        all_results.append({
            "query": query,
            "rank": result["rank"],
            "image": result["image"],
            "similarity": result["similarity"],
            "caption": result["caption"]
        })


# Save ALL search results to CSV 
if all_results:

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        SEARCH_CSV,
        index=False
    )


    # Save ALL search results to JSON
    json_data = {}

    for item in all_results:

        query_name = item["query"]

        if query_name not in json_data:

            json_data[query_name] = []

        json_data[query_name].append({
            "rank": item["rank"],
            "image": item["image"],
            "similarity": item["similarity"],
            "caption": item["caption"]
        })

    with open(
        SEARCH_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nAll search results saved:")

else:
     print("\nNo search queries were entered.")
     