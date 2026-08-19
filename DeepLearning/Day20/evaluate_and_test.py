from ultralytics import YOLO
import time
import os

DATA_YAML = "car_segmentation_dataset/data.yaml"
SCRATCH_MODEL = "scratch_model/best.pt"
PRETRAINED_MODEL = "pretrained_model/best.pt"
TEST_IMAGES = "test_images"

# VALIDATION
# scratch model evaluation
model = YOLO(SCRATCH_MODEL)
scratch_results = model.val(
    data=DATA_YAML,
    imgsz=320,
    batch=4,
    device="cpu"
)

# pretrained model evaluation
pretrained = YOLO(PRETRAINED_MODEL)
pretrained_results = pretrained.val(
    data=DATA_YAML,
    imgsz=320,
    batch=4,
    device="cpu"
)

# TESTING

os.makedirs("testing_output/scratch", exist_ok=True)
os.makedirs("testing_output/pretrained", exist_ok=True)

def test_model(model, model_name, output_folder):

    print(f"\nTesting {model_name}")

    image_files = [
        os.path.join(TEST_IMAGES, file)
        for file in os.listdir(TEST_IMAGES)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total_inference_time = 0

    for image in image_files:

        start_time = time.time()

        results = model.predict(
            source=image,
            imgsz=320,
            device="cpu",
            save=True,
            project=output_folder,
            name="predictions",
            exist_ok=True,
            verbose=False
        )

        inference_time = time.time() - start_time
        total_inference_time += inference_time

        print(
            f"{os.path.basename(image)}: "
            f"{inference_time * 1000:.2f} ms"
        )

    average_time = total_inference_time / len(image_files)

    print(
        f"\n{model_name} average inference time: "
        f"{average_time * 1000:.2f} ms/image"
    )

    return average_time


scratch_inference = test_model(
    model,
    "Scratch Model",
    "testing_output/scratch"
)

pretrained_inference = test_model(
    pretrained,
    "Pretrained Model",
    "testing_output/pretrained"
)
