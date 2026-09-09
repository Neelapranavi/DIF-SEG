# DIF-SEG

**Data-Driven Model for Medical Image Detection and Segmentation**

DIF-SEG is a Streamlit-based medical-image analysis project built around a real public benchmark rather than synthetic demo data. The current implementation uses the **ISIC 2016 Part 3B** dermoscopic lesion dataset for binary malignancy classification and lesion segmentation.

> **Research disclaimer:** DIF-SEG is an academic decision-support project. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## What the project does

1. Accepts a dermoscopic medical image.
2. Preprocesses the image for model inference.
3. Predicts benign vs. malignant using a trained binary classifier.
4. Segments the lesion using a trained U-Net.
5. Displays the predicted mask and allows it to be downloaded.
6. Calculates Dice and IoU when a ground-truth mask is supplied.
7. Provides classification metrics for held-out evaluation.
8. Provides an experimental diffusion-inspired augmentation component for research comparisons.

## Real dataset

The project uses the **ISIC 2016 Part 3B** training data from the International Skin Imaging Collaboration. The official ISIC documentation describes 900 dermoscopic lesion images, 900 paired segmentation masks, and 900 ground-truth malignancy labels (`benign` / `malignant`). The segmentation masks were created by expert clinicians, and the malignancy labels were obtained from expert consensus and pathology report information.

The official ISIC dataset page lists the 2016 data as CC-0. Always check the current official terms before redistributing or publishing derived material.

Official source: https://challenge.isic-archive.com/data/

## Project structure

```text
DIF-SEG/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── preprocessing/
├── detection/
├── segmentation/
├── diffusion/
│
├── evaluation/
│   ├── metrics.py
│   └── classification_metrics.py
│
├── scripts/
│   ├── download_isic2016.py
│   ├── prepare_isic2016.py
│   └── validate_isic2016.py
│
└── dataset/
    ├── README.md
    └── segmentation_dataset.py
```

Real medical data and trained model checkpoints are intentionally excluded from Git.

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
.venv\\Scripts\\activate
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

## Download the real medical dataset

Run:

```bash
python -m scripts.download_isic2016
python -m scripts.prepare_isic2016
python -m scripts.validate_isic2016
```

The first command downloads the official ISIC 2016 Part 3B training ZIP and ground-truth CSV. The second command extracts and organizes the cases into the folders used by DIF-SEG. The third command verifies that images, segmentation masks, and malignancy labels are correctly paired before training.

The prepared dataset is intentionally ignored by Git because it is large and should be obtained directly from the official source.

## Train the real models

### Segmentation

```bash
python -m segmentation.train
```

This trains the U-Net using the real ISIC lesion images and expert-created masks. The best checkpoint is saved as:

```text
models/unet_best.pt
```

### Detection

```bash
python -m detection.train
```

This trains the binary classifier using the same real ISIC cases, with `benign` mapped to `normal` and `malignant` mapped to `abnormal`. The best checkpoint is saved as:

```text
models/detection_best.pt
```

## Run the application

After both checkpoints exist:

```bash
streamlit run app.py
```

Then upload a dermoscopic lesion image and use the Detection, Segmentation, and Evaluation tabs.

## Evaluation

Do not report fabricated performance numbers. Train on a training split, evaluate on held-out data, and record the actual results.

For segmentation, DIF-SEG currently supports:

- Dice coefficient
- Intersection over Union (IoU)

For binary classification, `evaluation/classification_metrics.py` provides:

- Accuracy
- Precision
- Sensitivity / Recall
- Specificity
- F1-score
- Average Precision when probabilities are available
- Confusion-matrix counts

The official ISIC 2016 documentation also discusses sensitivity, specificity, accuracy, Jaccard, Dice, and average precision for its challenge tasks.

## Diffusion component

`diffusion/augmentation.py` contains a lightweight **diffusion-inspired Gaussian noise augmentation baseline**. It is not a trained DDPM and should not be presented as a validated generative diffusion model.

## Important limitations

- The current dataset is dermoscopic skin-lesion imagery, not CT/MRI/X-ray data.
- A benchmark dataset does not make the model clinically validated.
- Model performance depends on the training split, preprocessing, hyperparameters, and hardware.
- External validation on data from another institution is required before any claim of clinical generalization.
- Never upload identifiable patient data to a public repository or an untrusted deployment.
