# DIF-SEG

**Data-Driven Medical Image Detection and Segmentation**

DIF-SEG is a research-oriented medical image analysis project focused on preprocessing, image segmentation with U-Net, evaluation, and experimental diffusion-based data enhancement.

> **Important:** DIF-SEG is a research/educational decision-support prototype. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## Features

- Streamlit web interface
- Medical image upload and preprocessing
- U-Net segmentation inference
- Predicted binary mask visualization
- Mask download
- Dice and IoU evaluation utilities
- Experimental diffusion module for research workflows
- CPU/GPU inference support

## Project structure

```text
DIF-SEG/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── preprocessing/
│   └── preprocess.py
├── segmentation/
│   ├── __init__.py
│   └── unet.py
├── evaluation/
│   └── metrics.py
├── detection/
├── diffusion/
└── dataset/
```

Model checkpoints and datasets are intentionally excluded from Git because they can be large and may contain sensitive data.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Neelapranavi/DIF-SEG.git
cd DIF-SEG
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The Streamlit interface can run without a checkpoint, but segmentation inference requires a trained U-Net checkpoint at:

```text
models/unet_best.pt
```

## Dataset

Use a properly licensed medical imaging dataset with corresponding image/mask pairs. Keep the actual dataset outside Git when licensing, privacy, or repository size requires it.

## Evaluation

For labeled test data, report metrics such as:

- Dice coefficient
- Intersection over Union (IoU)
- Precision
- Recall
- F1-score

Only calculate and report metrics from actual test predictions and ground-truth labels.

## Research workflow

```text
Medical Image
     ↓
Preprocessing
     ↓
U-Net Segmentation
     ↓
Predicted Mask
     ↓
Evaluation
     ↓
Research Analysis
```

The diffusion component is experimental and should be treated as a research augmentation component rather than a validated clinical model.
