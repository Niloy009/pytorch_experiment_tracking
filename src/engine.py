"""
engine.py

This module contains functions for training and testing PyTorch models.
It provides modular steps for running a single training epoch, a single 
testing epoch, and a full multi-epoch training process with integrated 
experiment tracking via TensorBoard.

Key Features:
-------------
- `train_step`: Executes one epoch of training on a given dataset.
- `test_step`: Executes one evaluation pass on a test dataset.
- `train`: Runs multiple epochs of training and testing in sequence,
        logging results to TensorBoard.

Functionality includes:
- Support for custom models, optimizers, loss functions, and metrics.
- Automatic accuracy computation using torchmetrics.
- Logging of metrics and model graphs to TensorBoard.
- Device-agnostic execution (CPU/GPU).

Typical Usage:
--------------
    from torch.utils.tensorboard import SummaryWriter
    import torchmetrics

    writer = SummaryWriter(log_dir="runs/experiment1")
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    results = train(
        model=my_model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy=accuracy,
        epochs=10,
        device=torch.device("cuda"),
        writer=writer
    )

Author: Niloy Saha Roy
Created: 2025-08-14
"""

from typing import Dict, List, Tuple
import torch
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from tqdm.auto import tqdm

def train_step(model:torch.nn.Module,
            dataloader:torch.utils.data.DataLoader,
            loss_fn:torch.nn.Module,
            optimizer:torch.optim.Optimizer,
            accuracy:torchmetrics.classification.accuracy.Accuracy,
            device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch
    
    Turns a target PyTorch model to training model and then runs
    through all of the required training step (forward pass, 
    loss claculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to be minimized.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        accuracy: A torchmetric module to calculate accuracy.
        device: A target device to compute on (i.e. "cuda" or "cpu")

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    # Put the model in train mode.
    model.train()

    # Setup train loss and train accuracy
    train_loss, train_accuracy = 0, 0

    # Loop through the dataloader data batches
    for _, (X,y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward Pass
        y_logits = model(X)

        # 2. Claculate loss and accumulate loss
        loss = loss_fn(y_logits, y)
        # print(f"loss: \n {loss}")
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric accross all batches
        y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
        accuracy_value = accuracy(y_pred, y)
        # print(f"accuracy: \n {accuracy}")
        train_accuracy += accuracy_value.item()

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)

    return train_loss, train_accuracy

def test_step(model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            accuracy: torchmetrics.classification.accuracy.Accuracy,
            device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch
    
    Turns a target PyTorch model to "eval" mode and then performs
    forward pass on testing dataset and also calculate testing loss
    and testing accuracy.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        accuracy: A torchmetric module to calculate accuracy.
        device: A target device to compute on (i.e. "cuda" or "cpu")

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0112, 0.9343)
    """
    # Put the model in eval mode
    model.eval()

    # Setup the test loss and test accuracy
    test_loss, test_accuracy = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batchs
        for _, (X,y) in enumerate(dataloader):
            # Send data to a target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_logits, y)
            test_loss += loss.item()

            # Calculate accumulate accuracy
            y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            accuracy_value = accuracy(y_pred, y)
            test_accuracy += accuracy_value.item()

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_accuracy = test_accuracy / len(dataloader)

    return test_loss, test_accuracy


def train(model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        accuracy: torchmetrics.classification.accuracy.Accuracy,
        epochs: int,
        device: torch.device,
        writer: SummaryWriter) -> Dict[str, List]:
    """Trains and test a PyTorch model

    Passes a target PyTorch model through train_step() and test_step()
    functions for a number of epochs. training and testing the model in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.
    
    Stores metrics to specified writer log_dir if present.
    
    Args:
        model: A PyTorch model to be tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        accuracy: A torchmetric module to calculate accuracy.
        epochs: An integar indicating how many epochs to train for.
        device: A target device to compute on (i.e. "cuda" or "cpu").
        writer: A SummaryWriter() instance to log model results to.


    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]} 
        For example if training for epochs=2: 
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]} 
"""
    # Create empty results dictionary
    results = { "train_loss": [],
                "train_accuracy": [], 
                "test_loss": [], 
                "test_accuracy": []
}

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(model=model,
                                                dataloader=train_dataloader,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer,
                                                accuracy=accuracy,
                                                device=device)
        test_loss, test_accuracy = test_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            accuracy=accuracy,
                                            device=device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss: .4f} | "
            f"train_accuracy: {train_accuracy: .4f} | "
            f"test_loss: {test_loss: .4f} | "
            f"test_accuracy: {test_accuracy: .4f}"
        )
        # 5. update the results
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)

        #### New: Experiment tracking with tensorboard ####
        if writer:
            # Add training and testing loss metrics
            writer.add_scalars(main_tag="Loss",
                            tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                            global_step=epoch)
            # Add accuracy metrics
            writer.add_scalars(main_tag="Accuracy",
                            tag_scalar_dict={"train_accuracy": train_accuracy,
                                            "test_accuracy": test_accuracy},
                            global_step=epoch)
            # Add model graph to tensorboard
            writer.add_graph(model=model, input_to_model=torch.randn(32,3,224,224).to(device))

            # Close the writer
            writer.close()
        else:
            pass

        #### End: Experiment tracking with tensorboard ####
    return results
