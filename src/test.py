import random
from pathlib import Path
import torch
from src import model_builder, predictions

DATASET_ROOT = Path("data/pizza_steak_sushi_20")    # Path to the dataset root directory
BEST_MODEL_PATH = Path("artifacts/models/effnetb2_20pct_10ep.pth")  # Path to the best model state_dict
NUM_IMAGES_TO_PLOT = 4  # Number of images to plot predictions for
IMAGE_SIZE = (224, 224) # Image size for the model input
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Device to run the model on



def build_model(model_name: str) -> torch.nn.Module:
    """Return an EfficientNet model instance by name ('effnetb0' or 'effnetb2')."""
    if model_name == "effnetb0":
        return model_builder.create_effnetb0()
    if model_name == "effnetb2":
        return model_builder.create_effnetb2()
    raise ValueError(f"Unknown model: {model_name}")


def get_class_names_from_train(dir: Path):
    """Infer class names from subfolders of the train directory."""
    return sorted([p.name for p in dir.iterdir() if p.is_dir()])


def main():
    test_dir = DATASET_ROOT / "test"
    class_names = get_class_names_from_train(test_dir)
    if not class_names:
        raise RuntimeError(f"No class folders found under {test_dir}")

    # build the model and load the best state dict
    best_model = build_model("effnetb2").to(DEVICE)
    state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE)

    # Load the state dict into the model
    best_model.load_state_dict(state_dict)
    # Set the model to evaluation mode
    best_model.eval()

    # find test images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    test_images = [p for p in (test_dir).rglob("*") if p.suffix.lower() in exts]

    if not test_images:
        raise RuntimeError(f"No test images found under {test_dir}")

    k = min(NUM_IMAGES_TO_PLOT, len(test_images))
    sample_paths = random.sample(test_images, k=k)

    # Plot predictions for a sample of test images
    for image_path in sample_paths:
        predictions.pred_and_plot_image(
            model=best_model,
            image_path=str(image_path),
            class_names=class_names,
            image_size=IMAGE_SIZE,
            device=DEVICE,
        )

if __name__ == "__main__":
    main()
