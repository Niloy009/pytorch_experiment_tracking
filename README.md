# PyTorch Experiment Tracking (Learning Purpose)

## 📌 Overview
This is a **learning project** to understand how to track multiple experiments efficiently in PyTorch.  
It is based on the course [PyTorch for Deep Learning (Udemy)](https://www.udemy.com/share/107xb23@RUavMD6_EBgSB_soCutfJzyeDMpzQTEweXRu6-4gPHcovnM6C0jsxC_hFu5xLbAK0w==/).

The project uses **EfficientNet-B0** and **EfficientNet-B2** on the [Pizza, Steak, and Sushi dataset](https://github.com/mrdbourke/pytorch-deep-learning),  
which is a subset of the original [Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

It demonstrates:
- Experiment reproducibility  
- TensorBoard tracking  
- Automatic dataset preparation

## ✨ Features
- ✅ Dataset Download & Preprocessing (10% & 20% splits)
- ✅ EfficientNet-B0 and B2 (ImageNet-pretrained)
- ✅ Transfer Learning with Frozen Feature Extractor
- ✅ TensorBoard Logging (loss, accuracy, graph)
- ✅ Custom Classifier Heads for 3-class Output
- ✅ Reproducibility with Random Seeds
- ✅ Modular Codebase and CLI Support

## 🗂️ Project Structure
```
.
├── artifacts            # Saved models
│   └── models
├── data                 # Downloaded datasets
│   ├── pizza_steak_sushi_10
│   └── pizza_steak_sushi_20
├── LICENSE
├── README.md
├── runs                  # TensorBoard logs
│   └── 2025-Aug-14
└── src
    ├── data_module.py    # DataLoader creation
    ├── data_setup.py     # Dataset download and extraction
    ├── engine.py         # Train/test steps
    ├── __init__.py
    ├── model_builder.py  # EfficientNet model creation
    ├── predictions.py    # Single image prediction and visualization
    ├── train.py          # Main training orchestration (CLI + logging + experiment loop)
    └── utils.py          # Utility functions (saving, logging, plotting)

```

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Niloy009/pytorch_experiment_tracking.git
cd pytorch_experiment_tracking
```

### 2. Install Dependencies
```bash
pip install torch torchvision torchmetrics matplotlib requests tqdm tensorboard
```

## 🚀 Usage

### Train Models (Default Settings)
```bash
python train.py
```

### Custom Experiment Example
```bash
python train.py \
    --models effnetb0 effnetb2 \
    --splits 10 20 \
    --epochs 5 10 \
    --batch-size 32 \
    --lr 0.001 \
    --seed 42 \
    --artifacts-dir ./artifacts
```

## 📊 TensorBoard Logging
Start TensorBoard to monitor metrics:
```bash
tensorboard --logdir runs
```
Then navigate to [http://localhost:6006](http://localhost:6006)

## 🔍 Predict on Single Image
```python
from predictions import pred_and_plot_image
from model_builder import create_effnetb0
import torch

model = create_effnetb0()
model.load_state_dict(torch.load("path_to_model.pth"))
class_names = ["pizza", "steak", "sushi"]
pred_and_plot_image(model, class_names, "path_to_image.jpg")
```

## 📈 Tracked Metrics
- Train/Test Loss per Epoch
- Train/Test Accuracy per Epoch
- Model Graph Summary

## 🧠 Implementation Details
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Accuracy: torchmetrics.Accuracy
- Transforms: Model-specific (from torchvision weights)
- Device: Automatic CUDA/CPU selection
- Seeded Execution: `torch.manual_seed()`

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author
**Niloy Saha Roy**  
_Machine Learning & Deep Learning Engineer_ <br>
Email: niloysaha.887@gmail.com
