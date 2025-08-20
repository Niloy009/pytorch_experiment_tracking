"""
train .py

Training and experiment management for EfficientNet models using PyTorch.

This module provides a command-line interface and supporting utilities for running
reproducible experiments with EfficientNet models on custom datasets. It supports
configuration via YAML files and command-line overrides, manages data downloading
and preparation, constructs models and transforms, and handles training, evaluation,
and artifact saving for each experiment run.

Key features:
    - Flexible configuration: Combine YAML and CLI overrides for experiment settings.
    - Automated data handling: Download and prepare dataset splits as needed.
    - Model management: Build and train EfficientNet models with configurable hyperparameters.
    - Experiment tracking: Save model weights, configuration snapshots, and logs for each run.
    - Modular utilities: Includes helpers for logging, device selection, and directory management.

Typical usage involves running this script with a configuration file and optional
command-line overrides to launch a series of training and evaluation runs, with
results and artifacts saved for later analysis.

General usage:
    python -m src.train --config configs/default.yaml [--models effnetb0 effnetb2] [--data_pct 10 20] [--epochs 5 10] [--batch-size 32] [--lr 0.001] [--seed 42] [--data-root data] [--artifacts-dir artifacts] [-v|-vv]

Arguments in brackets are optional and can be used to override settings in the YAML config file.

Author: Niloy Saha Roy
Created: 2025-08-17
"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import yaml
import torch
from torch import nn
import torchvision
from torchmetrics.classification import Accuracy

from src import data_setup, data_module, utils, engine, model_builder


# Logging
def setup_logging(verbosity: int = 1) -> None:
    """Configures the logging level and format for the training script.

    Sets the logging verbosity based on the provided level and 
    applies a consistent format to log messages.

    Args:
        verbosity: The verbosity level (0=WARNING, 1=INFO, 2=DEBUG).
    """
    level = logging.INFO if verbosity == 1 else logging.DEBUG if verbosity > 1 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# -----------------------------
# Config dataclass
# -----------------------------
@dataclass(frozen=True)
class RunConfig:
    """Holds all configuration parameters for a training run.

    This dataclass encapsulates device, hyperparameters, dataset info, 
    and paths for a single experiment.
    """
    device: str
    batch_size: int
    learning_rate: float
    epochs_list: List[int]
    models: List[str]
    data_pct: List[int]
    seed: int
    data_root: Path
    artifacts_dir: Path
    datasets: Dict[str, Dict[str, str]]
    test_pct: int
    # keep the full, merged (effective) config for saving
    effective_cfg: Dict[str, Any]


# Utility helpers
def get_device(pref: str = "auto") -> str:
    """Selects the appropriate device for computation based on 
    user preference and hardware availability.

    Returns 'cuda' if requested or available, otherwise returns 'cpu'.

    Args:
        pref: Preferred device as a string ('auto', 'cuda', or 'cpu').

    Returns:
        The selected device as a string ('cuda' or 'cpu').
    """
    if pref == "cuda":
        return "cuda"
    if pref == "cpu":
        return "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Loads a YAML file and returns its contents as a dictionary.

    Reads the YAML file at the given path and parses it into a Python dictionary. 
    Returns an empty dictionary if the file is empty.

    Args:
        path: The path to the YAML file.

    Returns:
        A dictionary containing the parsed YAML data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dirs(*paths: Path) -> None:
    """Ensures that the specified directories exist, creating them if necessary.

    Iterates over the provided paths and creates each directory and 
    its parents if they do not already exist.

    Args:
        *paths: One or more Path objects representing directories to create.
    """
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _merge_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merges command-line argument overrides into a base configuration dictionary.

    For each supported config key, replaces the value with the corresponding CLI argument 
    if provided, applying any necessary transformations.

    Args:
        cfg: The base configuration dictionary loaded from YAML.
        args: The argparse.Namespace containing parsed CLI arguments.

    Returns:
        A new dictionary with CLI overrides applied to the base configuration.
    """

    merged = dict(cfg)  # shallow copy

    def _set_if_provided(key: str, cli_value, transform=None):
        if cli_value is None:
            return
        merged[key] = transform(cli_value) if transform else cli_value

    # Set arguments if provided
    _set_if_provided("batch_size", args.batch_size)
    _set_if_provided("learning_rate", args.lr)
    _set_if_provided("seed", args.seed)
    _set_if_provided("models", args.models)
    _set_if_provided("data_pct", args.data_pct, transform=lambda v: [int(x) for x in v])
    _set_if_provided("epochs_list", args.epochs, transform=lambda v: [int(x) for x in v])
    _set_if_provided("data_root", str(args.data_root) if args.data_root else None)
    _set_if_provided("artifacts_dir", str(args.artifacts_dir) if args.artifacts_dir else None)
    return merged


def save_effective_config(effective_cfg: Dict[str, Any], target_dir: Path) -> None:
    """Saves the effective configuration dictionary to a YAML file in the specified directory.

    Attempts to write the configuration to 'config_used.yaml' and logs the outcome.

    Args:
        effective_cfg: The configuration dictionary to save.
        target_dir: The directory where the YAML file will be written.
    """

    try:
        out = target_dir / "config_used.yaml"
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(effective_cfg, f, sort_keys=False)
        logging.info("Saved effective config to %s", out)
    except OSError as e:
        logging.warning("Could not save effective config: %s", e)


# Data utilities
def download_splits(cfg: RunConfig) -> Dict[int, Path]:
    """Downloads and prepares dataset splits as specified in the configuration.

    Iterates over all requested splits, downloads each dataset if necessary, 
    and returns a mapping from split percentage to local path.

    Args:
        cfg: The RunConfig object containing dataset and split information.

    Returns:
        A dictionary mapping each split percentage to its prepared local Path.
    """

    out: Dict[int, Path] = {}
    for s in cfg.data_pct:
        s_key = str(s)
        if s_key not in cfg.datasets:
            raise KeyError(
                f"data percent {s} missing in config.datasets. "
                f"Available: {list(cfg.datasets.keys())}"
            )
        url = cfg.datasets[s_key]["url"]
        dest = cfg.datasets[s_key]["destination"]
        out[s] = data_setup.download_data(source=url, destination=dest)
        logging.info("Prepared data %s%% at %s", s, out[s])
    return out


def build_transform_for_model(model_name: str):
    """Returns the appropriate image transformation pipeline for a given EfficientNet model.

    Selects the default torchvision transforms for the specified model name.

    Args:
        model_name: The name of the EfficientNet model ('effnetb0' or 'effnetb2').

    Returns:
        The torchvision transform pipeline for the specified model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if model_name == "effnetb0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    elif model_name == "effnetb2":
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return weights.transforms()


def build_model(model_name: str) -> torch.nn.Module:
    """Constructs and returns an EfficientNet model instance based on the given model name.

    Selects the appropriate model builder for 'effnetb0' or 'effnetb2' 
    and raises an error for unknown names.

    Args:
        model_name: The name of the EfficientNet model to build ('effnetb0' or 'effnetb2').

    Returns:
        An instance of the specified EfficientNet model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if model_name == "effnetb0":
        return model_builder.create_effnetb0()
    if model_name == "effnetb2":
        return model_builder.create_effnetb2()
    raise ValueError(f"Unknown model: {model_name}")


def experiment_tag(pct: int, model_name: str, epochs: int) -> str:
    """Generates a unique experiment tag string based on split, model, and epochs.

    Combines the data percentage, model name, and number of epochs into a standardized tag.

    Args:
        pct: The data percentage.
        model_name: The name of the model.
        epochs: The number of training epochs.

    Returns:
        A string representing the experiment tag.
    """
    return f"{pct}pct/{model_name}/{epochs}_epochs"


def save_path(artifacts_dir: Path, pct: int, model_name: str, epochs: int) -> Path:
    """Generates a unique file path for saving model artifacts for a specific experiment run.

    Creates file name based on model, data percentage, and epochs, 
    ensuring the directory exists.

    Args:
        artifacts_dir: The root directory for storing artifacts.
        pct: The data percentage.
        model_name: The name of the model.
        epochs: The number of training epochs.

    Returns:
        The full Path to the file where model weights should be saved.
    """
    fname = f"{model_name}_{pct}pct_{epochs}ep"
    # store each run in its own folder (nice place for config + weights)
    run_dir = artifacts_dir / "models"
    return run_dir / f"{fname}.pth"


# Core runner
def run(cfg: RunConfig) -> None:  # sourcery skip: convert-to-enumerate
    """Executes a full training and evaluation cycle for all experiment configurations.

    Iterates over all combinations of data splits, models, and epochs, training and 
    evaluating each, and saving results and configurations.

    Args:
        cfg: The RunConfig object containing all experiment parameters.
    """
    logging.info("Using device: %s", cfg.device)
    utils.set_seeds(cfg.seed)

    # Prepare data
    data_pct_paths = download_splits(cfg)
    test_pct = cfg.test_pct
    if test_pct not in data_pct_paths:
        raise KeyError(
            f"test_split={test_pct} not among downloaded splits {list(data_pct_paths)}. "
            "Ensure it is included in config.splits."
        )
    test_dir = data_pct_paths[test_pct] / "test"  # shared test set

    exp_counter = 0
    for split, model_name, epochs in itertools.product(cfg.data_pct, cfg.models, cfg.epochs_list):
        exp_counter += 1
        tag = experiment_tag(split, model_name, epochs)
        logging.info("======== Experiment %d: %s ========", exp_counter, tag)

        transform = build_transform_for_model(model_name)

        train_dir = data_pct_paths[split] / "train"
        train_dl, test_dl, class_names = data_module.create_dataloaders(
            train_dir=str(train_dir),
            test_dir=str(test_dir),
            transform=transform,
            batch_size=cfg.batch_size,
            num_workers=os.cpu_count(),
        )
        logging.info("Batches | train=%d | test=%d | classes=%s",
                    len(train_dl), len(test_dl), class_names)

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

        # Save weights and the effective config snapshot
        path = save_path(cfg.artifacts_dir, split, model_name, epochs)
        utils.save_model(model=model, target_dir=str(path.parent), model_name=path.name)
        save_effective_config(cfg.effective_cfg, target_dir=path.parent)

        # Report best epoch (by test accuracy)
        test_accuracies = results["test_accuracy"]
        best_ep = max(range(len(test_accuracies)),
                    key=lambda i: test_accuracies[i])
        logging.info("Best epoch: %d | test_acc: %.4f", best_ep + 1,
                    test_accuracies[best_ep])


# CLI
def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the training script.

    Defines and processes all CLI options for configuring the training run, including config file, 
    model selection, data splits, and other overrides.

    Returns:
        An argparse.Namespace containing the parsed arguments.
    """
    p = argparse.ArgumentParser(description="Train EfficientNet models on Pizza/Steak/Sushi.")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                help="Path to YAML config file.")
    # Optional overrides (CLI > YAML)
    p.add_argument("--models", nargs="+",
                help="Override models list, e.g., --models effnetb0 effnetb2")
    p.add_argument("--data_pct", nargs="+", type=int,
                help="Override dataset percentage, e.g., --data_pct 10 20")
    p.add_argument("--epochs", nargs="+", type=int,
                help="Override epochs list, e.g., --epochs 5 10")
    p.add_argument("--batch-size", type=int,
                help="Override batch size")
    p.add_argument("--lr", type=float, help="Override learning rate")
    p.add_argument("--seed", type=int, help="Override RNG seed")
    p.add_argument("--data-root", type=Path, help="Override data root")
    p.add_argument("--artifacts-dir", type=Path, help="Override artifacts dir")
    p.add_argument("-v", "--verbose", action="count", default=1,
                help="Increase verbosity (-v=INFO, -vv=DEBUG)")
    return p.parse_args()


def main() -> None:
    """Entry point for the training script.

    Parses command-line arguments, loads and merges configuration, 
    prepares directories, and starts the training run.
    """
    args = parse_args()
    setup_logging(args.verbose)

    # 1) Load YAML
    base_cfg = _load_yaml(args.config)

    # 2) Merge CLI overrides
    merged = _merge_cli_overrides(base_cfg, args)

    # 3) Normalize / defaults
    device = get_device(merged.get("device", "auto"))
    batch_size = int(merged.get("batch_size", 32))
    learning_rate = float(merged.get("learning_rate", 1e-3))
    epochs_list = [int(x) for x in merged.get("epochs_list", [5, 10])]
    models = list(merged.get("models", ["effnetb0", "effnetb2"]))
    data_pct = [int(x) for x in merged.get("data_pct", [10, 20])]
    seed = int(merged.get("seed", 42))
    data_root = Path(merged.get("data_root", "data"))
    artifacts_dir = Path(merged.get("artifacts_dir", "artifacts"))

    datasets = merged.get("datasets", {})
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Config is missing 'datasets' mapping with URLs/destinations.")

    test_pct = int(merged.get("test_pct", 10))

    # Ensure dirs exist
    _ensure_dirs(data_root, artifacts_dir)

    # 4) Build effective config snapshot (what the run actually uses)
    effective_cfg = {
        "device": device,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs_list": epochs_list,
        "models": models,
        "data_pct": data_pct,
        "seed": seed,
        "data_root": str(data_root),
        "artifacts_dir": str(artifacts_dir),
        "datasets": datasets,
        "test_pct": test_pct,
        "source_config_file": str(args.config.resolve()),
    }

    # 5) Create RunConfig and go
    cfg = RunConfig(
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs_list=epochs_list,
        models=models,
        data_pct=data_pct,
        seed=seed,
        data_root=data_root,
        artifacts_dir=artifacts_dir,
        datasets=datasets,
        test_pct=test_pct,
        effective_cfg=effective_cfg,
    )
    run(cfg)


if __name__ == "__main__":
    main()
