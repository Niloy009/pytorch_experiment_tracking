"""
data_setup.py

This module handles to download the data and unzip it to a specified location.

- Downloading and extracting datasets from remote sources.

Functions:
----------
- download_data: Downloads and extracts a zipped dataset to a specified location.

Author: Niloy Saha Roy
Created: 2025-08-14
"""
import os
import zipfile
from pathlib import Path
import requests


def download_data(source:str, destination:str, remove_source:bool = True)->Path:
    """Download a ziped dataset from source and unzip to destination
    Args:
        source: The source path where the data will download from.
        destination: The destination path where the data will download and unzip to.
        remove_source: Whether the source remove or not after download.

    Returns:
        pathlib.Path to downloaded data.
    """
    # Setup data path
    data_path = Path("../data/")
    image_path = data_path / destination # images from a subset of classes from the Food101 dataset

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping re-download.")
    else:
        print(f"[INFO] Did not find {image_path}, downloading it...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        target_file = Path(source).name # Extract the filename from the source URL
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source, timeout=30)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file}...")
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
            print(f"[INFO] Data downloaded and unzipped to {image_path}")
    return image_path
