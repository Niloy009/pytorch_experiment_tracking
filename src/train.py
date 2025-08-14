"""
train.py

Train EfficientNet models on Pizza/Steak/Sushi dataset splits with experiment tracking.

This module orchestrates a matrix of experiments across:
- dataset splits (10%, 20%)
- model variants (EfficientNet-B0, EfficientNet-B2)
- epoch counts (e.g., 5, 10)

It handles data download, dataloader creation, model construction, training,
TensorBoard logging, and artifact saving. Designed for reproducibility and
CLI-driven configuration.

Typical usage:
    python train.py --models effnetb0 effnetb2 --splits 10 20 --epochs 5 10 --batch-size 32

Requirements:
- Local modules: data_setup, data_module, engine, model_builder, utils
- PyTorch, torchvision, torchmetrics, TensorBoard

Author: Niloy Saha Roy
Created: 2025-08-14
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
import torchvision
from torchmetrics.classification import Accuracy

from . import data_setup, data_module, utils, engine, model_builder

# import data_setup
# import data_module
# import utils
# import engine
# import model_builder


def setup_logging(verbosity: int = 1) -> None:
    """
    Configure the root logger with a simple, timestamped format.

    Args:
        verbosity: Verbosity level where
            0 = WARNING, 1 = INFO (default), 2+ = DEBUG.

    Returns:
        None. Configures global logging state.

    Example:
        setup_logging(verbosity=2)
    """
    level = logging.INFO if verbosity == 1 else logging.DEBUG if verbosity > 1 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass(frozen=True)
class RunConfig:
    """
    Immutable configuration bundle for an experiment.

    Attributes:
        device: String name of compute device ("cuda" or "cpu").
        batch_size: Mini-batch size for both train and test dataloaders.
        learning_rate: Initial learning rate for the optimizer.
        epochs_list: List of epoch counts to iterate over (e.g., [5, 10]).
        models: Model names to train and test (e.g., ["effnetb0", "effnetb2"]).
        splits: Dataset split percentages to use (e.g., [10, 20]).
        seed: Global seed for reproducibility.
        data_root: Base directory under which datasets are stored.
        artifacts_dir: Directory where model weights and artifacts are saved.
    """
    device: str
    batch_size: int
    learning_rate: float
    epochs_list: List[int]
    models: List[str]
    splits: List[int]
    seed: int
    data_root: Path
    artifacts_dir: Path


def get_device() -> str:
    """
    Return the preferred compute device based on CUDA availability.

    Returns:
        "cuda" if a CUDA device is available, otherwise "cpu".

    Example:
        device = get_device()
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def download_splits(cfg: RunConfig) -> Dict[int, Path]:
    """
    Download and extract the required dataset splits.

    For convenience, this function uses the 10% dataset's **test** split for all runs,
    matching the behavior in your original script.

    Args:
        cfg: The experiment configuration.

    Returns:
        Mapping of split percentage -> local Path of the extracted dataset root.

    Raises:
        KeyError: If cfg.splits includes an unsupported percentage.
    """
    # Define URLs for dataset splits
    urls = {
        10: "https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip",
        20: "https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi_20_percent.zip",
    }
    # Download each requested split and return a mapping
    out: Dict[int, Path] = {}
    for s in cfg.splits:
        if s not in urls:
            raise KeyError(f"Unsupported split {s}%. Supported: {list(urls)}.")
        dest = f"pizza_steak_sushi_{s}"
        out[s] = data_setup.download_data(source=urls[s], destination=dest)
        logging.info("Prepared split %s%% at %s", s, out[s])
    return out


def build_transform_for_model(model_name: str):
    """
    Return the torchvision preprocessing pipeline corresponding to the model's default weights.

    Args:
        model_name: One of {"effnetb0", "effnetb2"}.

    Returns:
        A torchvision transforms pipeline (Compose) appropriate for the models input size
        and normalization

    Raises:
        ValueError: If an unknown model name is provided.

    Example:
        transform = build_transform_for_model("effnetb2")
    """
    if model_name == "effnetb0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    elif model_name == "effnetb2":
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return weights.transforms()


def build_model(model_name: str) -> torch.nn.Module:
    """
    Construct a model instance by name using the local model factory.

    Args:
        model_name: One of {"effnetb0", "effnetb2"}.

    Returns:
        A PyTorch nn.Module instance placed on the appropriate device.

    Raises:
        ValueError: If an unknown model name is provided.

    Example:
        model = build_model("effnetb0")
    """
    if model_name == "effnetb0":
        return model_builder.create_effnetb0()
    if model_name == "effnetb2":
        return model_builder.create_effnetb2()
    raise ValueError(f"Unknown model: {model_name}")


def experiment_tag(split: int, model_name: str, epochs: int) -> str:
    """
    Create a hierarchical tag string for organizing TensorBoard runs.

    Args:
        split: Dataset split percentage (e.g., 10).
        model_name: Model name (e.g., "effnetb0").
        epochs: Number of training epochs.

    Returns:
        A tag string like "10pct/effnetb0/5_epochs".

    Example:
        tag = experiment_tag(10, "effnetb0", 5)
    """
    return f"{split}pct/{model_name}/{epochs}_epochs"


def save_path(artifacts_dir: Path, split: int, model_name: str, epochs: int) -> Path:
    """
    Build a timestamped model file path within the artifacts directory.

    Args:
        artifacts_dir: Root directory for saved artifacts.
        split: Dataset split percentage (e.g., 10).
        model_name: Model name string (e.g., "effnetb2").
        epochs: Number of training epochs used.

    Returns:
        A Path pointing to a file like: artifacts/models/20250101-120000_effnetb2_10pct_5ep.pth

    Example:
        path = save_path(Path("artifacts"), 20, "effnetb0", 10)
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{ts}_{model_name}_{split}pct_{epochs}ep.pth"
    return artifacts_dir / "models" / fname


def run(cfg: RunConfig) -> None:
    """
    Execute the full experiment over splits_models_epochs.

    This function:
    1) Downloads required dataset splits.
    2) Builds per-model transforms.
    3) Creates train/test DataLoaders.
    4) Constructs model, loss, optimizer, and accuracy metric.
    5) Trains and evaluates each configuration.
    6) Logs metrics and saves checkpoints.

    Args:
        cfg: Frozen configuration specifying experiment knobs and paths.

    Returns:
        None. Side effects include TensorBoard logs and saved model weights.
    """
    logging.info("Using device: %s", cfg.device)
    utils.set_seeds(cfg.seed)

    # cuDNN performance knobs
    torch.backends.cudnn.benchmark = (cfg.device == "cuda")
    torch.backends.cudnn.deterministic = False

    # Prepare data
    split_paths = download_splits(cfg)
    test_dir = split_paths[10] / "test"  # single test set for all runs

    exp_counter = 0
    for split, model_name, epochs in itertools.product(cfg.splits, cfg.models, cfg.epochs_list):
        exp_counter += 1
        tag = experiment_tag(split, model_name, epochs)
        logging.info("======== Experiment %d: %s ========", exp_counter, tag)

        transform = build_transform_for_model(model_name)

        train_dir = split_paths[split] / "train"
        train_dl, test_dl, class_names = data_module.create_dataloaders(
            train_dir=str(train_dir),
            test_dir=str(test_dir),
            transform=transform,
            batch_size=cfg.batch_size,
            num_workers=os.cpu_count(),
        )
        logging.info("Batches | train=%d | test=%d | classes=%s", len(train_dl), len(test_dl), class_names)

        model = build_model(model_name)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        metric = Accuracy(task="multiclass", num_classes=len(class_names)).to(cfg.device)

        writer = utils.create_writer(
            experiment_name=f"data_{split}_percent",
            model_name=model_name,
            extra=f"{epochs}_epochs"
        )

        results = engine.train(
            model=model,
            train_dataloader=train_dl,
            test_dataloader=test_dl,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy=metric,
            epochs=epochs,
            device=cfg.device,
            writer=writer
        )

        # Save latest weights (timestamped)
        path = save_path(cfg.artifacts_dir, split, model_name, epochs)
        utils.save_model(model=model, target_dir=str(path.parent), model_name=path.name)

        # Report best epoch (by test accuracy) to logs
        best_ep = max(range(len(results["test_accuracy"])), key=lambda i: results["test_accuracy"][i])
        logging.info("Best epoch: %d | test_acc: %.4f", best_ep + 1, results["test_accuracy"][best_ep])


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for experiment configuration.

    CLI flags:
        --models       One or more model names (effnetb0, effnetb2).
        --splits       One or more dataset splits (10, 20).
        --epochs       One or more epoch counts (e.g., 5 10).
        --batch-size   Mini-batch size (default: 32).
        --lr           Learning rate (default: 1e-3).
        --seed         random seed (default: 42).
        --data-root    Base path for datasets (default: ./data).
        --artifacts-dir Path for saved models and outputs (default: ./artifacts).
        -v/--verbose   Increase verbosity (-v=INFO, -vv=DEBUG).

    Returns:
        argparse.Namespace of parsed arguments.

    Example:
        args = parse_args()
    """
    p = argparse.ArgumentParser(description="Train EfficientNet models on Pizza/Steak/Sushi.")
    p.add_argument("--models", nargs="+", default=["effnetb0", "effnetb2"],
                help="Models to run. Choices: effnetb0 effnetb2")
    p.add_argument("--splits", nargs="+", type=int, default=[10, 20],
                help="Dataset splits to use (10, 20).")
    p.add_argument("--epochs", nargs="+", type=int, default=[5, 10],
                help="Epoch counts to run.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    p.add_argument("-v", "--verbose", action="count", default=1,
                help="Increase verbosity (-v, -vv).")
    return p.parse_args()


def main() -> None:
    """
    Entrypoint: parse CLI, set up logging, construct config, run experiments.

    Returns:
        None. Side effects include running training jobs and writing artifacts.
    """
    args = parse_args()
    setup_logging(args.verbose)

    # Ensure directories exist
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Create the run configuration
    cfg = RunConfig(
        device=get_device(),
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs_list=args.epochs,
        models=args.models,
        splits=args.splits,
        seed=args.seed,
        data_root=args.data_root,
        artifacts_dir=args.artifacts_dir,
    )
    # Run the experiment
    run(cfg)


if __name__ == "__main__":
    main()
