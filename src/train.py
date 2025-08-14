"""
train.py

This module orchestrates multiple deep learning experiments for image classification 
using EfficientNet architectures (B0 and B2) on different dataset splits (10% and 20%) 
of the Pizza, Steak, and Sushi dataset. It automates dataset preparation, model creation, 
training, evaluation, experiment tracking, and model saving.

Key Features:
-------------
- Downloads and prepares multiple dataset splits (10% and 20%).
- Creates DataLoaders with pretrained EfficientNet transformation pipelines.
- Runs experiments across combinations of:
    * Model architectures: EfficientNet-B0, EfficientNet-B2
    * Dataset sizes: 10%, 20%
    * Training epochs: [5, 10]
- Tracks experiments using TensorBoard SummaryWriter.
- Saves trained models for future use.

Workflow:
---------
1. Download datasets from predefined sources.
2. Create DataLoaders for each dataset split.
3. Iterate through combinations of dataset splits, model types, and training epochs.
4. For each combination:
    - Initialize model, loss function, optimizer, and accuracy metric.
    - Train and evaluate the model.
    - Log results with TensorBoard.
    - Save trained model weights to disk.

Functions Used (from other modules):
------------------------------------
- data_setup.download_data(): Downloads and extracts datasets.
- data_module.create_dataloaders(): Creates train/test DataLoaders.
- model_builder.create_effnetb0(), create_effnetb2(): Builds EfficientNet models.
- engine.train(): Trains and evaluates the model per experiment.
- utils.create_writer(): Sets up TensorBoard logging.
- utils.save_model(): Saves the trained model to disk.

Usage:
------
    python train.py

Author: Niloy Saha Roy
Created: 2025-08-14
"""

import torch
from torch import nn
import torchvision
from torchmetrics.classification.accuracy import Accuracy
from . import data_setup, data_module, utils, engine, model_builder

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup the source and destination paths for the datasets
SOURCE_10 = "https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip"
SOURCE_20 = "https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi_20_percent.zip"
DESTINATION_10 = "pizza_steak_sushi_10"
DESTINATION_20 = "pizza_steak_sushi_20"

# Setup the hyperparameters
BATCH_SIZE = 32 # Setup the Batch Size
NUM_EPOCHS = [5,10] # Create epoch list
LEARNING_RATE = 0.001 # Setup the learning rate

# Create model list (need to create a new mdoel for each experiment)
models = ['effnetb0', 'effnetb2']



# Download 10 percent and 20 percent datasets
data_10_percent_path = data_setup.download_data(source=SOURCE_10, destination=DESTINATION_10)
data_20_percent_path = data_setup.download_data(source=SOURCE_20, destination=DESTINATION_20)

# Get the train and test path
train_10_dir = data_10_percent_path / 'train'
train_20_dir = data_20_percent_path / 'train'
test_10_dir = data_10_percent_path / 'test'


# Setup pretrained weights
effnet_b0_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

# Get the transform of the pretrained model best weights
effnet_b0_transform = effnet_b0_weights.transforms()
print(f'[INFO] Creating Transform from pretrained model best weights: {effnet_b0_transform}')

# Create the dataloaders for 10%
train_dataloader_10, test_datalaoder, class_names = data_module.create_dataloaders(
    train_dir=train_10_dir,
    test_dir=test_10_dir,
    transform=effnet_b0_transform,
    batch_size=BATCH_SIZE)

# Create the dataloaders for 20%
train_dataloader_20, test_dataloader, class_names = data_module.create_dataloaders(
    train_dir=train_20_dir,
    test_dir=test_10_dir,
    transform=effnet_b0_transform,
    batch_size=BATCH_SIZE
    )

# Print the number of batches and class names
print(f'[INFO] Number of batche size {BATCH_SIZE} in 10% train data: {len(train_dataloader_10)}')
print(f'[INFO] Number of batche size {BATCH_SIZE} in 20% train data: {len(train_dataloader_20)}')
print(f'[INFO] Number of batche size {BATCH_SIZE} in 10% test data: {len(test_dataloader)}')
print(f'[INFO] Class names: {class_names} & length of the class: {len(class_names)}')


# Create a DataLoaders dictionary
train_dataloaders = {'data_10_percent': train_dataloader_10,
                    'data_20_percent':  train_dataloader_20}

# Set Seeds
utils.set_seeds()

# Keep track of experiment numbers
EXPERIMENT_NUMBER = 0

# Loop through each DataLoaders
for dataloader_name, train_dataloader in train_dataloaders.items():
    # Loop through the epochs
    for epochs in NUM_EPOCHS:
        # Loop through the model name and Create a new model instance
        for model_name in models:

            EXPERIMENT_NUMBER += 1

            # Print out the info
            print(f"[INFO] Experminet Number: {EXPERIMENT_NUMBER}")
            print(f"[INFO] Model: {model_name}")
            print(f"[INFO] DataLoader: {dataloader_name}")
            print(f"[INFO] Number of Epochs: {epochs}")

            # Select and Create the model
            if model_name == 'effnetb0':
                model = model_builder.create_effnetb0()
            else:
                model = model_builder.create_effnetb2()

            # Create a new loss, optimizer and accuracy for every model
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
            accuracy = Accuracy(task='multiclass', num_classes=len(class_names)).to(DEVICE)

            # Train target model with target dataloader and track experiment
            engine.train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        accuracy=accuracy,
                        epochs=epochs,
                        device=DEVICE,
                        writer=utils.create_writer(experiment_name=dataloader_name,
                                            model_name=model_name, extra=f"{epochs}_epochs"))

            # Save the model to file so we can import it later if needed
            SAVE_FILE_PATH = f"07_{model_name}_{dataloader_name}_{epochs}_epochs.pth"
            utils.save_model(model=model, target_dir='models', model_name=SAVE_FILE_PATH)
            print("-"*50 + "\n")
