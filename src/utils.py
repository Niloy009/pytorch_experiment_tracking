"""
utils.py

This module contains utility functions to support PyTorch model training, 
saving, experiment tracking, and visualization.

Key Features:
-------------
- Reproducibility: Set deterministic random seeds for CPU and GPU operations.
- Model Persistence: Save model weights to a specified directory.
- Experiment Tracking: Create TensorBoard SummaryWriter instances with organized log directory structure.
- Training Visualization: Plot training and validation loss/accuracy curves.

Functions:
----------
- set_seeds(seed=42):
    Sets the random seeds for CPU and CUDA operations to ensure reproducibility.
- save_model(model, target_dir, model_name):
    Saves a PyTorch model's state_dict to disk with a specified file name.
- create_writer(experiment_name, model_name, extra=None):
    Creates a TensorBoard SummaryWriter instance with timestamped log directory.
- plot_loss_curves(results):
    Plots loss and accuracy curves from a results dictionary.

Usage Example:
--------------
    from utils import set_seeds, save_model, create_writer, plot_loss_curves

    set_seeds(42)
    writer = create_writer("experiment_1", "resnet50", "10_epochs")
    save_model(model, "models", "resnet50.pth")
    plot_loss_curves(results)

Author: Niloy Saha Roy
Created: 2025-08-14
"""


import os
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """

    # set the seed for the general torch operation
    torch.manual_seed(seed)

    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory
    
    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include either ".pth" or ".pt" as the file extension.

    Example Usage:
        save_model(model=model_1, 
                target_dir='models', 
                model_name='tiny_vgg_model.pth')    
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pth' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def create_writer(experiment_name: str, model_name: str, extra: str = None) -> SummaryWriter:
    """Create a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir
    
    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra

    Where timestamp is current date in YYYY-MM-DD format

    Args:
        experiment_name (str): Name of the experiment.
        model_name (str): Name of the model
        extra (str, optional): Anything extra to add to the directory.

    Returns:
        SummaryWriter(): Instance of a writer saving to the specific log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent", 
                            model_name="effnetb2",
                            extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    # Get timestamp of current date in reverse order (YYYY-MM-DD)
    timestamp = datetime.now().strftime("%Y-%b-%d")

    if extra:
        # create log directory path
        log_dir = os.path.join('../runs', timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join('../runs', timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter saving to {log_dir}")
    return SummaryWriter(log_dir=log_dir)





def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
            "train_accuracy": [...],
            "test_loss": [...],
            "test_accuracy": [...]}
    """
    # Get the loss values from the results dictionary (training and testing)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values from the results dictionary (training & testing)
    accuracy = results["train_accuracy"]
    test_accuracy = results["test_accuracy"]

    # Get the epochs
    epochs = range(len(results["train_loss"]))

    # Setup the plot
    plt.figure(figsize=(15,7))

    # plot the loss
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss")
    plt.title("Loss Curves")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()

    # plot the loss
    plt.subplot(1,2,2)
    plt.plot(epochs, accuracy, label="Train Accuracy")
    plt.plot(epochs, test_accuracy, label="Test Accuracy")
    plt.title("Accuracy Curves")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
