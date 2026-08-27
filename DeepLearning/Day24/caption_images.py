import os
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


input_folder = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day24\input"
output_folder = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day24\results"

os.makedirs(output_folder, exist_ok=True)

# Load BLIP 

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


# Generate captions

results = []
image_extensions = (".jpg", ".jpeg", ".png", ".webp")

for filename in os.listdir(input_folder):

    if not filename.lower().endswith(image_extensions):
        continue

    image_path = os.path.join(input_folder, filename)

    try:
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        output = model.generate(
            **inputs,
            max_new_tokens=30
        )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        results.append({
            "image": filename,
            "caption": caption
        })

        print(f"{filename} → {caption}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")


# Save captions
df = pd.DataFrame(results)

output_path = os.path.join(
    output_folder,
    "image_captions.csv"
)

df.to_csv(output_path, index=False)

print("\nDone!")