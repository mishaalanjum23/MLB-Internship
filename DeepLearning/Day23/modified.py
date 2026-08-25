from PIL import Image, ImageEnhance
import os

input_path = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day23\input\cycle.jpg"
output_dir = r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\DeepLearning\Day23\input"

image = Image.open(input_path)

# 1. Resized version
resized = image.resize((image.width // 2, image.height // 2))
resized.save(os.path.join(output_dir, "cycle_resized.jpg"))

# 2. Cropped version
crop = image.crop((
    int(image.width * 0.1),
    int(image.height * 0.1),
    int(image.width * 0.9),
    int(image.height * 0.9)
))
crop.save(os.path.join(output_dir, "cycle_cropped.jpg"))

# 3. Brightness-changed version
enhancer = ImageEnhance.Brightness(image)
brighter = enhancer.enhance(1.5)
brighter.save(os.path.join(output_dir, "cycle_bright.jpg"))

print("Modified images created!")