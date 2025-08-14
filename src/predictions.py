"""
predictions.py

This module provides utility functions for making predictions with trained PyTorch models
and visualizing the results on individual images.

Key Features:
-------------
- Loads and preprocesses images for model inference.
- Supports custom or default image transformations.
- Runs predictions on CPU or GPU automatically based on availability.
- Displays the predicted class label and probability directly on the image.

Functions:
----------
- pred_and_plot_image(model, class_names, image_path, image_size=(224, 224),
                    transform=None, device=DEVICE):
    Predicts the class of a single image using a trained PyTorch model, 
    applies the specified transformation (or ImageNet defaults), and 
    visualizes the prediction along with its probability.

Usage Example:
--------------
    from predict import pred_and_plot_image

    pred_and_plot_image(
        model=my_trained_model,
        class_names=["cat", "dog", "horse"],
        image_path="test_images/dog.jpg"
    )

Author: Niloy Saha Roy
Created: 2025-08-14
"""

from typing import List, Tuple
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Predict on a target image with a target model
def pred_and_plot_image(model: torch.nn.Module,
                        class_names: List[str],
                        image_path: str,
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device = DEVICE,
):
    """Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_path (str): Filepath to target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
    """

    # Open image
    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    ### Predict on image ###

    # Make sure the model is on the target device
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)
