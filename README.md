# DIF-SEG

**Data-Driven Medical Image Detection and Segmentation**

DIF-SEG is a Streamlit-based research prototype for medical image analysis. It combines binary abnormality detection, U-Net segmentation, segmentation evaluation, reusable preprocessing, and an experimental diffusion-inspired augmentation component.

> **Important:** DIF-SEG is an educational/research decision-support prototype. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## Features

- Streamlit web interface
- Binary normal/abnormal detection baseline
- U-Net image segmentation
- Dice and IoU evaluation
- CPU/GPU inference support
- Reusable preprocessing utilities
- Experimental diffusion-inspired noise augmentation
- Downloadable predicted segmentation masks

## Project structure

```text
DIF-SEG/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── preprocessing/
│   ├── __init__.py
│   └── preprocess.py
├── detection/
│   ├── __init__.py
│   ├── model.py
│   └── train.py
├── segmentation/
│   ├── __init__.py
│   ├── unet.py
│   └── train.py
├── diffusion/
│   ├── __init__.py
│   └── augmentation.py
├── evaluation/
│   └── metrics.py
└── dataset/
    ├── README.md
    └── segmentation_dataset.py
```

Model checkpoints and real medical datasets are intentionally excluded from Git because model files can be large and datasets may contain sensitive or restricted information.

## Setup

### 1. Clone

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

### 4. Start Streamlit

```bash
streamlit run app.py
```

## Segmentation dataset

Create paired image and mask folders:

```text
dataset/
├── images/
│   ├── image_001.png
│   └── image_002.png
└── masks/
    ├── image_001.png
    └── image_002.png
```

Image and mask filenames must match. Then train the U-Net:

```bash
python -m segmentation.train
```

The best checkpoint is saved locally as:

```text
models/unet_best.pt
```

## Detection dataset

For binary detection, organize images as:

```text
dataset/
└── detection/
    ├── normal/
    │   ├── image_001.png
    │   └── image_002.png
    └── abnormal/
        ├── image_101.png
        └── image_102.png
```

Train the CNN baseline with:

```bash
python -m detection.train
```

The best checkpoint is saved locally as:

```text
models/detection_best.pt
```

## Evaluation

For labeled test data, DIF-SEG currently provides:

- Dice coefficient
- Intersection over Union (IoU)

Additional metrics such as precision, recall, accuracy, and F1-score can be reported when the corresponding test setup is appropriate. Report metrics only from actual predictions and ground-truth labels.

## Diffusion component

`diffusion/augmentation.py` contains a lightweight **diffusion-inspired Gaussian noise augmentation baseline**. It is intentionally not presented as a trained DDPM or a validated generative diffusion model. It can be used for controlled research experiments comparing training with and without noise augmentation.

## Research workflow

```text
                  Medical Image
                       │
                  Preprocessing
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Detection            Segmentation
       Normal/Abnormal           U-Net
             │                   │
             └─────────┬─────────┘
                       ▼
                 Evaluation
                 Dice / IoU
                       │
                       ▼
                Research Analysis
```

## Data and safety

Use properly licensed datasets. Do not commit private, identifiable, or restricted medical data to a public repository. This project is intended for academic and research use only.
